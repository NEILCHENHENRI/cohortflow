from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from framework.session import SteerableSession
from framework.task_delta import predicate_ref_maps
from framework.graph import ExecutionNode, NodeKind
from presentation.models import (
    AgentMetadataView, ConceptEntryView, CriterionEvidenceView, EntityCollectionView, EntityDetailView,
    ExecutionEvidenceView, ExecutionStateView, GraphEdgeView, GraphGroupView, GraphNodeView, GraphView,
    MetricView, NodeDetailView, SectionKind, SectionView,
)

@runtime_checkable
class PatientDataProvider(Protocol):
    def patient_detail(self, patient_id: int) -> dict[str, Any]: ...

class CohortPresentationAdapter:
    def __init__(self, session: SteerableSession, patient_provider: PatientDataProvider | None=None) -> None:
        self.session=session
        self.agent=self.session.domain_agent
        self.patient_provider=patient_provider

    def metadata(self) -> AgentMetadataView:
        return AgentMetadataView(
            "cohort_curation",
            "CohortFlow",
            "Turn natural-language eligibility criteria into an inspectable cohort, then revise it with mid-run corrections.",
            "cohort_curation",
            "🩺",
        )

    def graph_view(self) -> GraphView:
        return self._build_graph_view()

    def _build_graph_view(self) -> GraphView:
        state = self._state()
        evidence = self.execution_evidence()
        criterion_by_id = {
            item.requirement_id: item for item in evidence.criteria
        }
        # CohortPlanner rejects any task without a canonical Boolean AST
        # before a session ever reaches this state, so `logic` is guaranteed.
        logic = state.task.logic
        _, predicate_ref_by_id = predicate_ref_maps(state.task)

        nodes: list[GraphNodeView] = []
        edges: list[GraphEdgeView] = []
        # Multiple facts often route through the same table (e.g. every
        # condition's membership is computed against diagnoses_icd). One
        # shared node per unique table keeps this from turning into a wall
        # of duplicate boxes.
        source_node_ids: dict[str, str] = {}

        def add_expression(expression, path: str) -> tuple[str, ...]:
            if expression.kind == "predicate":
                rid = expression.predicate_id or path
                criterion = criterion_by_id.get(rid)
                display_ref = predicate_ref_by_id.get(rid, rid)
                node_id = f"logic:criterion:{rid}"
                if criterion is None:
                    label = expression.text or rid
                    metric = None
                    subtitle = "Unresolved predicate"
                    mode = "unchanged"
                    runtime_id = "aggregate:cohort"
                    concept_entries: tuple[ConceptEntryView, ...] = ()
                else:
                    label = criterion.label
                    metric = f"{criterion.patient_count} patients"
                    subtitle = expression.text or "Clinical predicate"
                    mode = self._requirement_execution_mode(rid)
                    runtime_id = self._runtime_node_for_requirement(rid)
                    concept_entries = criterion.concept_entries

                # The criterion is the natural-language keyword/request.  It
                # sits *after* the database/fact-resolution evidence and
                # immediately before the Boolean logic tree, making the
                # graph read as one coherent explanation:
                #
                #   database -> resolved code/fact -> requested keyword
                #            -> AND/OR/NOT logic -> final cohort
                #
                # This is intentionally a presentation/provenance view, not
                # a literal trace of runtime call order.
                nodes.append(
                    GraphNodeView(
                        node_id=node_id,
                        label=label,
                        status="completed",
                        category="criterion",
                        subtitle=subtitle,
                        tooltip=expression.text or label,
                        execution_mode=mode,
                        primary_metric=metric,
                        detail_ref=runtime_id,
                        metadata={"runtime_node_id": runtime_id, "structure_ref": display_ref},
                    )
                )

                # Resolved codes/facts feed INTO the natural-language
                # criterion. A multi-code resolution therefore converges on
                # one keyword node rather than fanning out after it.  This is
                # both easier to read and guarantees a single clean edge from
                # each predicate into its Boolean parent gate.
                if not concept_entries:
                    return (node_id,)

                for index, entry in enumerate(concept_entries):
                    concept_node_id = f"logic:concept:{rid}:{index}"
                    # The first source table is normally the dictionary used
                    # for terminology lookup (e.g. d_icd_diagnoses). Keep it
                    # as a provenance badge on the code. Remaining table(s)
                    # are the patient-level EHR sources and are rendered as
                    # the beginning of this branch.
                    dictionary_tables: tuple[str, ...] = ()
                    ehr_tables: tuple[str, ...] = entry.source_tables
                    if len(entry.source_tables) > 1:
                        dictionary_tables = entry.source_tables[:1]
                        ehr_tables = entry.source_tables[1:]

                    nodes.append(
                        GraphNodeView(
                            node_id=concept_node_id,
                            label=entry.code,
                            status="completed",
                            category="concept",
                            subtitle=entry.vocabulary_id,
                            tooltip=entry.display_name,
                            execution_mode=mode,
                            badges=tuple(f"📖 {table}" for table in dictionary_tables),
                            detail_ref=runtime_id,
                            metadata={"runtime_node_id": runtime_id},
                        )
                    )
                    edges.append(GraphEdgeView(concept_node_id, node_id))

                    for table in ehr_tables:
                        # One shared node per unique patient-level source
                        # table. Shared tables can fan out to several codes,
                        # after which each code converges on its own keyword.
                        source_id = source_node_ids.get(table)
                        if source_id is None:
                            source_id = f"logic:source:{table}"
                            source_node_ids[table] = source_id
                            nodes.append(
                                GraphNodeView(
                                    node_id=source_id,
                                    label=table,
                                    status="completed",
                                    category="source",
                                    subtitle="mimic_hosp.duckdb",
                                    tooltip=f"EHR table queried: {table}",
                                    execution_mode="unchanged",
                                    detail_ref=runtime_id,
                                    metadata={"runtime_node_id": runtime_id},
                                )
                            )
                        edges.append(GraphEdgeView(source_id, concept_node_id))

                return (node_id,)

            node_id = f"logic:op:{expression.node_id or path}"
            label = expression.kind.upper()
            subtitle = {
                "and": "All branches required",
                "or": "Any branch sufficient",
                "not": "Exclude matching patients",
            }[expression.kind]
            nodes.append(
                GraphNodeView(
                    node_id=node_id,
                    label=label,
                    status="completed",
                    category="logic",
                    subtitle=subtitle,
                    tooltip=subtitle,
                    execution_mode=self._execution_mode("aggregate:cohort"),
                    detail_ref="aggregate:cohort",
                    metadata={"runtime_node_id": "aggregate:cohort"},
                )
            )
            for index, child in enumerate(expression.children):
                child_exit_ids = add_expression(child, f"{path}.{index}")
                for child_exit_id in child_exit_ids:
                    edges.append(GraphEdgeView(child_exit_id, node_id))
            return (node_id,)

        root_ids = add_expression(logic, "root")
        result = self._logic_result_node(evidence, state)
        nodes.append(result)
        for root_id in root_ids:
            edges.append(GraphEdgeView(root_id, result.node_id))

        # A shared source/concept node reached via multiple sibling
        # criteria (e.g. two conditions both routing through the same
        # diagnoses_icd table) gets a structurally-identical edge appended
        # once per criterion -- exit-node dedup only happens within a
        # single criterion's own chain, not across siblings feeding the
        # same gate. Harmless to a viewer but pure visual clutter, so
        # collapse duplicate (source, target) pairs before returning.
        seen_edges: set[tuple[str, str]] = set()
        deduped_edges = []
        for edge in edges:
            key = (edge.source, edge.target)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            deduped_edges.append(edge)

        return GraphView(
            tuple(nodes),
            tuple(deduped_edges),
            state.version,
            "Cohort graph",
            self._legend(),
            (),
            "Database evidence, resolved clinical codes, your request keywords, and Boolean logic -- ending in the final cohort.",
        )

    @staticmethod
    def _logic_result_node(evidence: ExecutionEvidenceView, state) -> GraphNodeView:
        return GraphNodeView(
            node_id="logic:result",
            label="Final cohort",
            status=("completed" if evidence.summary_available else state.status.value),
            category="output",
            subtitle="Patients satisfying the full Boolean expression",
            primary_metric=(
                "Not produced"
                if evidence.final_count is None
                else f"{evidence.final_count} patients"
            ),
            detail_ref="deliver:summary",
            metadata={"runtime_node_id": "deliver:summary"},
        )

    def execution_state(self) -> ExecutionStateView:
        state = self._state()
        node_counts: dict[str, int] = {}
        for node in state.graph.nodes.values():
            key = node.status.value
            node_counts[key] = node_counts.get(key, 0) + 1
        pending = state.pending_interaction
        summary = None
        if self.session.artifacts.contains("cohort_summary"):
            value = self.session.artifacts.get("cohort_summary").value
            if isinstance(value, dict):
                summary = dict(value)
        return ExecutionStateView(
            version=state.version,
            status=state.status.value,
            node_counts=node_counts,
            artifact_ids=self.session.artifacts.ids(),
            pending_kind=(None if pending is None else pending.kind.value),
            pending_question=(None if pending is None else pending.question),
            summary=summary,
        )

    def execution_evidence(self) -> ExecutionEvidenceView:
        state = self._state()
        execution = self.execution_state()
        by_id = {r.requirement_id: r for r in state.task.requirements}
        criteria = []
        evidence_by_requirement: dict[str, dict[str, Any]] = {}
        for aid in self.session.artifacts.ids():
            if not aid.startswith("evidence:"):
                continue
            value = self.session.artifacts.get(aid).value
            if isinstance(value, dict):
                evidence_by_requirement[str(value.get("requirement_id", ""))] = value

        # Always show current requirements, even before they have evidence.
        for requirement in state.task.requirements:
            value = evidence_by_requirement.get(requirement.requirement_id, {})
            resolved = value.get("resolved_concept") or {}
            # The criterion is the extracted keyword (what the request asked
            # for). The resolved concept is a separate "fact" surfaced via
            # its own concept nodes below. Letting the fact's name overwrite
            # the criterion's own label here would collapse two genuinely
            # different pieces of information into one, and hide the
            # original keyword the moment resolution succeeds.
            label = self._requirement_display_label(requirement, state.task.logic)
            criteria.append(CriterionEvidenceView(
                requirement.requirement_id,
                str(label),
                str(value.get("kind", requirement.kind)),
                str(value.get("polarity", requirement.polarity)),
                int(value.get("patient_count", 0)),
                tuple(int(x) for x in value.get("patient_ids", [])),
                None if resolved.get("concept_id") is None else str(resolved["concept_id"]),
                resolved.get("concept_name"),
                resolved.get("vocabulary_id"),
                resolved.get("resolution_source"),
                tuple(
                    ConceptEntryView(
                        str(entry["code"]),
                        str(entry["display_name"]),
                        str(entry["vocabulary_id"]),
                        tuple(str(t) for t in entry.get("source_tables", [])),
                    )
                    for entry in resolved.get("entries", [])
                ),
            ))

        raw = (
            self.session.artifacts.get("cohort_evidence").value
            if self.session.artifacts.contains("cohort_evidence")
            else {}
        )
        rows = tuple(
            {
                "patient_id": int(pid),
                "included": bool(details.get("included")),
                "matched_include": ", ".join(details.get("matched_include_requirements", [])),
                "matched_exclude": ", ".join(details.get("matched_exclude_requirements", [])),
            }
            for pid, details in sorted(
                (raw.get("patient_membership") or {}).items(),
                key=lambda item: int(item[0]),
            )
        )
        summary = execution.summary or {}
        final_count = raw.get("result_count")
        if final_count is None:
            final_count = summary.get("cohort_size")
        final_ids = raw.get("result_patient_ids")
        if final_ids is None:
            final_ids = summary.get("sample_patient_ids", [])

        if execution.pending_question:
            message = "Waiting for confirmation: " + execution.pending_question
        elif execution.status == "running":
            message = "Execution is in progress. Evidence will appear as nodes complete."
        elif execution.status == "failed":
            message = "Execution failed. Partial evidence remains available below."
        elif not execution.summary_available:
            message = "No final cohort has been produced yet. Showing currently available execution evidence."
        else:
            message = "Execution completed and final cohort evidence is available."

        return ExecutionEvidenceView(
            operator=(
                self._render_logic_for_display(state.task.logic, criteria)
                if state.task.logic is not None
                else str(
                    raw.get(
                        "include_operator",
                        state.task.metadata.get("include_operator", "and"),
                    )
                ).upper()
            ),
            final_count=(None if final_count is None else int(final_count)),
            final_patient_ids=tuple(int(x) for x in final_ids),
            criteria=tuple(sorted(criteria, key=lambda item: item.requirement_id)),
            membership_rows=rows,
            status=execution.status,
            status_message=message,
            summary_available=execution.summary_available,
        )

    @staticmethod
    def _render_logic_for_display(logic, criteria) -> str:
        labels = {item.requirement_id: item.label for item in criteria}

        def render(node) -> str:
            if node.kind == "predicate":
                rid = node.predicate_id or ""
                return labels.get(rid, node.text or rid)
            if node.kind == "not":
                return f"NOT ({render(node.children[0])})"
            joiner = f" {node.kind.upper()} "
            return "(" + joiner.join(render(child) for child in node.children) + ")"

        return render(logic)

    @staticmethod
    def _requirement_display_label(requirement, logic=None) -> str:
        payload = requirement.payload if isinstance(requirement.payload, dict) else {}
        term = payload.get("term")
        if isinstance(term, str) and term.strip():
            return term.strip()
        claim = payload.get("claim")
        if isinstance(claim, str) and claim.strip():
            return claim.strip()
        if logic is not None:
            for node in logic.leaves():
                if (
                    node.kind == "predicate"
                    and node.predicate_id == requirement.requirement_id
                    and isinstance(node.text, str)
                    and node.text.strip()
                ):
                    return node.text.strip()
        # Last-resort deterministic labels for tasks that do not carry leaf text.
        if requirement.kind == "patient":
            attribute = payload.get("attribute", "patient attribute")
            operator = payload.get("operator")
            value = payload.get("value")
            parts = [str(attribute)]
            if operator is not None:
                parts.append(str(operator))
            if value is not None:
                parts.append(str(value))
            return " ".join(parts)
        if requirement.kind == "admission":
            if payload.get("admission_type"):
                count = payload.get("count_min")
                prefix = f"at least {count} " if count is not None else ""
                return f"{prefix}{payload['admission_type']} admission" + ("s" if count and int(count) != 1 else "")
            if payload.get("count_min") is not None:
                return f"at least {payload['count_min']} hospital admissions"
            if payload.get("hospital_expire_flag") == 0:
                return "survived hospitalization"
        return requirement.kind.replace("_", " ")

    def _concept_node_view(self, node_id: str) -> NodeDetailView:
        # node_id format: logic:concept:{requirement_id}:{index}
        remainder = node_id[len("logic:concept:"):]
        rid, _, index_text = remainder.rpartition(":")
        evidence = self.execution_evidence()
        criterion = next(
            (c for c in evidence.criteria if c.requirement_id == rid), None,
        )
        entry = None
        if criterion is not None and index_text.isdigit():
            index = int(index_text)
            if 0 <= index < len(criterion.concept_entries):
                entry = criterion.concept_entries[index]
        if entry is None:
            raise KeyError(f"unknown concept node: {node_id}")

        sections = [
            SectionView(
                "concept_detail",
                "Resolved concept",
                SectionKind.KEY_VALUE,
                {
                    "Code": entry.code,
                    "Name": entry.display_name,
                    "Vocabulary": entry.vocabulary_id,
                    "Source tables": ", ".join(entry.source_tables) or "—",
                },
            ),
        ]
        if criterion is not None:
            sections.append(SectionView(
                "criterion_context",
                "Criterion",
                SectionKind.KEY_VALUE,
                {
                    "Criterion keyword": criterion.label,
                    "Patients matched by this criterion": str(criterion.patient_count),
                },
            ))
        return NodeDetailView(
            node_id=node_id,
            title=entry.code,
            description=entry.display_name,
            status="completed",
            category="concept",
            sections=tuple(sections),
        )

    def _source_node_view(self, node_id: str) -> NodeDetailView:
        table = node_id[len("logic:source:"):]
        evidence = self.execution_evidence()
        referencing = [
            criterion.label
            for criterion in evidence.criteria
            for entry in criterion.concept_entries
            if table in entry.source_tables
        ]
        return NodeDetailView(
            node_id=node_id,
            title=table,
            description=f"Raw MIMIC-IV hosp-module table: {table}",
            status="completed",
            category="source",
            sections=(
                SectionView(
                    "source_detail",
                    "Data source",
                    SectionKind.KEY_VALUE,
                    {"Table": table, "Database": "mimic_hosp.duckdb"},
                ),
                SectionView(
                    "referenced_by",
                    "Used by criteria",
                    SectionKind.TEXT,
                    ", ".join(referencing) or "—",
                ),
            ),
        )

    def node_view(self,node_id:str)->NodeDetailView:
        state=self._state()
        # Concept ("fact") and source ("database") nodes are synthesized
        # for the LOGIC view only -- they have no entry of their own in
        # state.graph.nodes, so mapping them to their parent criterion's
        # runtime node (as logic: nodes normally do below) would just show
        # the criterion's own generic detail again, with nothing about the
        # specific code/name or table actually clicked.
        if node_id.startswith("logic:concept:"):
            return self._concept_node_view(node_id)
        if node_id.startswith("logic:source:"):
            return self._source_node_view(node_id)
        # Projection nodes map to runtime nodes.
        if node_id.startswith("logic:"):
            mapping={n.node_id:n.metadata.get("runtime_node_id") for n in self._build_graph_view().nodes}
            node_id=str(mapping.get(node_id) or node_id)
        if node_id not in state.graph.nodes: raise KeyError(f"unknown graph node: {node_id}")
        node=state.graph.nodes[node_id]
        sections=[SectionView("dependencies","Dependencies",SectionKind.KEY_VALUE,{"Depends on":", ".join(sorted(state.graph.parents(node_id))) or "None","Inputs":", ".join(node.inputs) or "None","Outputs":", ".join(node.outputs) or "None"}),SectionView("configuration","Configuration",SectionKind.JSON,node.config)]
        if node_id == "deliver:summary":
            # cohort_summary only keeps a 10-ID sample. The full membership
            # set is already computed (it backs the Evidence tab's own
            # patient list), so surface it here too rather than making the
            # user hunt for it in a different tab.
            evidence = self.execution_evidence()
            sections.append(SectionView(
                "final_patient_ids",
                f"Full patient list ({len(evidence.final_patient_ids)})",
                SectionKind.ENTITY_LIST,
                [{"patient_id": pid} for pid in evidence.final_patient_ids],
                description="Every patient satisfying the full Boolean expression.",
            ))
        for output_id in node.outputs:
            if self.session.artifacts.contains(output_id):
                artifact=self.session.artifacts.get(output_id)
                value=artifact.value
                preview=value
                if isinstance(value,(set,frozenset,list,tuple)) and len(value)>50: preview=list(sorted(value))[:50]
                sections.append(SectionView(f"artifact:{output_id}",f"Artifact: {output_id}",SectionKind.JSON,preview,description="Preview. Large collections are truncated."))
        if node.error: sections.append(SectionView("error","Execution error",SectionKind.WARNING,node.error,True))
        return NodeDetailView(node_id,self._friendly_label(node),self._node_description(node),node.status.value,node.operator_name if node.kind is NodeKind.COMPUTATION else "input",(MetricView("Status",node.status.value),MetricView("Execution mode",self._execution_mode(node_id))),tuple(sections),self._collections_for_node(node))

    def entity_view(self,collection_id:str,entity_id:str)->EntityDetailView:
        if not collection_id.startswith("patients:"): raise KeyError(f"unknown entity collection: {collection_id}")
        patient_id=int(entity_id)
        if self.patient_provider is None: return EntityDetailView(str(patient_id),f"Patient {patient_id}","No patient-detail provider is configured.",{"Patient ID":patient_id})
        data=self.patient_provider.patient_detail(patient_id)
        sections=tuple(SectionView(sid,title,SectionKind.TABLE,rows) for sid,title,rows in (("conditions","Conditions",data.get("conditions",[])),("medications","Medications",data.get("medications",[])),("visits","Visits",data.get("visits",[]))) if rows)
        return EntityDetailView(str(patient_id),f"Patient {patient_id}","Information from the configured cohort database",dict(data.get("demographics",{})),sections)

    def _collections_for_node(self,node:ExecutionNode)->tuple[EntityCollectionView,...]:
        out=[]
        for oid in node.outputs:
            if not self.session.artifacts.contains(oid): continue
            value=self.session.artifacts.get(oid).value
            if not (oid=="cohort_patient_ids" or oid.startswith("patients:")) or not isinstance(value,(set,frozenset,list,tuple)): continue
            ids=sorted(int(x) for x in value); rows=tuple({"patient_id":x} for x in ids)
            out.append(EntityCollectionView(f"patients:{oid}","Final cohort patients" if oid=="cohort_patient_ids" else "Patients matching this criterion","patient",("patient_id",),rows,len(rows),"patient_id","Patients contained in this node's output artifact."))
        return tuple(out)

    def _runtime_node_for_requirement(self,rid:str)->str:
        for nid,node in self._state().graph.nodes.items():
            if nid.startswith("evaluate:requirement:") and node.config.get("requirement_id")==rid: return nid
        return "aggregate:cohort"
    def _requirement_execution_mode(self,rid:str)->str: return self._execution_mode(self._runtime_node_for_requirement(rid))
    def _execution_mode(self,nid:str)->str:
        report=self._state().last_reuse_report
        if report is None:return "unchanged"
        if nid in report.reused_nodes:return "reused"
        if nid in report.added_nodes:return "new"
        if nid in report.invalidated_nodes:return "recomputed"
        return "unchanged"
    @staticmethod
    def _friendly_label(node:ExecutionNode)->str:
        if node.node_id=="input:task":return "Cohort task"
        if node.node_id=="aggregate:cohort":return "Boolean aggregation" if node.config.get("logic") else str(node.config.get("include_operator","and")).upper()+" aggregation"
        if node.node_id=="validate:cohort":return "Validate cohort"
        if node.node_id=="deliver:summary":return "Final cohort summary"
        if node.node_id.startswith("input:requirement:"):return f"{str(node.config.get('kind','requirement')).title()} criterion"
        if node.node_id.startswith("evaluate:requirement:"):return str(node.config.get("requirement_id") or "Evaluate criterion")
        return node.operator_name or node.node_id
    @staticmethod
    def _node_description(node:ExecutionNode)->str:return str(node.config.get("requirement_id") or node.operator_name or node.node_id)
    @staticmethod
    def _legend()->dict[str,str]:return {"reused":"Reused from prior version","new":"Added by latest change","recomputed":"Invalidated and rerun","unchanged":"Unchanged execution"}
    def _state(self):
        if self.session.state is None:raise RuntimeError("cohort agent has not been initialized")
        return self.session.state
