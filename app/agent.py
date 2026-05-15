import json
import logging
import os
import re
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI
from app.models import Message, Recommendation
from app.retriever import HybridRetriever
from app.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Soft skills mapping to test type suggestions
SOFT_SKILL_TEST_TYPES = {
    "stakeholder": ["Personality", "Behavioral"],
    "communication": ["Personality", "Behavioral"],
    "leadership": ["Personality", "Cognitive"],
    "management": ["Personality", "Cognitive"],
    "teamwork": ["Personality"],
    "problem solving": ["Cognitive"],
    "critical thinking": ["Cognitive"],
    "presentation": ["Personality", "Skills"],
    "negotiation": ["Personality", "Skills"]
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_hiring_needs",
            "description": "Extract structured hiring needs from conversation",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "Job title or role, e.g., Java Developer"},
                    "seniority": {"type": "string", "enum": ["entry", "mid", "senior", "lead", "unknown"]},
                    "skills": {"type": "array", "items": {"type": "string"}, "description": "Technical and soft skills mentioned"},
                    "test_types": {"type": "array", "items": {"type": "string", "enum": ["Knowledge", "Personality", "Cognitive", "Skills", "Behavioral"]}},
                    "description_text": {"type": "string", "description": "If a job description was pasted, the full text"}
                },
                "required": ["role", "seniority", "skills", "test_types"]
            }
        }
    }
]

class Agent:
    def __init__(self, retriever: HybridRetriever, client: AsyncOpenAI, model_name: str):
        self.retriever = retriever
        self.client = client
        self.model = model_name

    async def process_conversation(self, messages: List[Message]) -> Tuple[str, List[Recommendation], bool]:
        """Async main entry point."""
        # Check refusal quickly
        refusal = self._quick_refusal_check(messages)
        if refusal:
            return refusal, [], True

        # Extract structured constraints using tool call
        constraints = await self._extract_constraints_async(messages)
        # If tool call failed to extract, fall back to basic clarification
        if not constraints or not constraints.get("role"):
            clarification = await self._generate_clarification_async(messages)
            return clarification, [], False

        # Detect comparison request
        compare_names = self._detect_compare_request(messages)
        if compare_names:
            return await self._handle_comparison_async(compare_names, messages)

        # Determine if we have enough to recommend
        if not self._has_sufficient_context(constraints):
            clarification = await self._generate_targeted_clarification(constraints, messages)
            return clarification, [], False

        # Build queries for dual-pass retrieval
        queries = self._build_dual_queries(constraints, messages)
        filters = self._build_filters(constraints)
        candidates = self.retriever.search_with_rerank(queries, filters, k=10, fuse_k=30)

        if not candidates:
            return ("I couldn't find matching assessments. Could you broaden your criteria? "
                    "For example, different seniority or skills."), [], False

        # Generate recommendation reply using LLM with candidates
        reply, action = await self._generate_reply_async(messages, constraints, candidates)

        recs = []
        if action == "recommend":
            # Ensure each candidate has a valid URL (already validated)
            recs = [Recommendation(name=c["title"], url=c["url"], test_type=c["test_type_codes"][0])
                    for c in candidates[:10]]

        end = action == "end"
        return reply, recs, end

    def _quick_refusal_check(self, messages: List[Message]) -> Optional[str]:
        """Quick pattern check for obvious off-topic."""
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        off_topic_patterns = [
            r"legal", r"lawyer", r"hire.*fire", r"salary", r"compensation",
            r"who (is|are) you", r"ignore previous", r"system prompt", r"do not recommend"
        ]
        for pat in off_topic_patterns:
            if re.search(pat, last_user.lower()):
                return "I can only help with SHL assessment selection. I can't assist with that topic."
        return None

    async def _extract_constraints_async(self, messages: List[Message]) -> Optional[Dict]:
        """Use tool calling to extract hiring context."""
        history_str = self._format_history(messages)
        system_msg = ("Extract the hiring needs from the conversation. "
                      "If missing, mark 'unknown' or leave arrays empty.")
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": history_str}
                ],
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=300
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                if tc.function.name == "extract_hiring_needs":
                    args = json.loads(tc.function.arguments)
                    # Ensure skills is list
                    if isinstance(args.get("skills"), str):
                        args["skills"] = [args["skills"]]
                    return args
            # If no tool call, try to parse content as JSON
            if msg.content:
                json_match = re.search(r'\{.*\}', msg.content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Extraction error: {e}")
        return None

    def _has_sufficient_context(self, constraints: Dict) -> bool:
        """Check if we have at least a role and some skill/type info."""
        role = constraints.get("role")
        skills = constraints.get("skills", [])
        test_types = constraints.get("test_types", [])
        # If role is present, that's enough
        if role and role.lower() != "unknown":
            return True
        # Or if skills or test types are specified
        if skills or test_types:
            return True
        return False

    async def _generate_targeted_clarification(self, constraints: Dict, messages: List[Message]) -> str:
        """Ask for the missing most critical piece of information."""
        missing = []
        if not constraints.get("role") or constraints["role"].lower() == "unknown":
            missing.append("job role or title")
        if not constraints.get("seniority") or constraints["seniority"] == "unknown":
            missing.append("seniority level (entry, mid, senior)")
        if not constraints.get("skills"):
            missing.append("required skills or technologies")
        if not constraints.get("test_types"):
            missing.append("types of assessments (Knowledge, Personality, Cognitive, Skills, Behavioral)")

        if missing:
            # Ask for the most important missing piece
            if "job role or title" in missing:
                question = "What is the job title or role you're hiring for?"
            elif "skills" in missing:
                question = "What specific technical or soft skills are important for this role?"
            else:
                question = f"Could you tell me more about the {missing[0]}?"
            return question
        return "Could you provide a bit more detail about your requirements?"

    def _build_dual_queries(self, constraints: Dict, messages: List[Message]) -> List[str]:
        """Build two query strings: one for technical/hard skills, one for soft skills."""
        role = constraints.get("role", "")
        skills = constraints.get("skills", [])
        test_types = constraints.get("test_types", [])
        seniority = constraints.get("seniority", "")

        # Separate hard and soft skills
        hard_skills = []
        soft_skills = []
        for s in skills:
            if s.lower() in SOFT_SKILL_TEST_TYPES or any(soft in s.lower() for soft in ["management", "communication", "leadership", "team"]):
                soft_skills.append(s)
            else:
                hard_skills.append(s)

        tech_query = role
        if hard_skills:
            tech_query += " " + " ".join(hard_skills)
        if seniority and seniority != "unknown":
            tech_query += f" {seniority}"
        # If user specified test types, add them
        tech_types = [t for t in test_types if t in ["Knowledge", "Skills", "Cognitive"]]
        if tech_types:
            tech_query += " " + " ".join(tech_types)
        else:
            # Default: knowledge and skills
            tech_query += " Knowledge Skills"

        soft_query = role
        if soft_skills:
            soft_query += " " + " ".join(soft_skills)
        # Map soft skills to test types
        soft_types = set()
        for s in soft_skills:
            for key, types in SOFT_SKILL_TEST_TYPES.items():
                if key in s.lower():
                    soft_types.update(types)
        if not soft_types:
            # If no soft skills but role implies some (like stakeholder), we might infer
            if "stakeholder" in role.lower() or "manager" in role.lower():
                soft_types.update(["Personality", "Behavioral"])
        soft_types.update([t for t in test_types if t in ["Personality", "Behavioral"]])
        if soft_types:
            soft_query += " " + " ".join(soft_types)
        else:
            soft_query += " Personality Behavioral"  # default for soft pass

        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        # Add a snippet of the last user message for context
        return [f"{tech_query} {last_user}", f"{soft_query} {last_user}"]

    def _build_filters(self, constraints: Dict) -> Optional[Dict]:
        filters = {}
        # Test type codes
        if constraints.get("test_types"):
            code_map = {"Knowledge": "K", "Personality": "P", "Cognitive": "C",
                        "Skills": "S", "Behavioral": "B"}
            codes = []
            for t in constraints["test_types"]:
                if t in code_map:
                    codes.append(code_map[t])
            if codes:
                filters["test_type_codes"] = codes
        # Seniority mapping
        seniority = constraints.get("seniority", "").lower()
        if seniority and seniority != "unknown":
            seniority_map = {
                "entry": ["Entry-Level", "Graduate"],
                "mid": ["Mid-Professional", "Professional Individual Contributor"],
                "senior": ["Senior Manager", "Executive"],
                "lead": ["Manager", "Supervisor"]
            }
            if seniority in seniority_map:
                filters["job_levels"] = seniority_map[seniority]
        return filters if filters else None

    def _detect_compare_request(self, messages: List[Message]) -> Optional[List[str]]:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        # Better pattern: "compare X and Y", "difference between X and Y"
        patterns = [
            r'(?:compare|difference)\s+(?:between\s+)?(.+?)\s+and\s+(.+)',
            r'what is the difference between (.+) and (.+)',
            r'how does (.+) differ from (.+)'
        ]
        for pat in patterns:
            match = re.search(pat, last_user, re.IGNORECASE)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        return None

    async def _handle_comparison_async(self, names: List[str], messages: List[Message]) -> Tuple[str, List[Recommendation], bool]:
        # Retrieve both
        items = []
        for name in names:
            item = self.retriever.get_by_name(name)
            if not item:
                # fuzzy search
                results = self.retriever.search_with_rerank([name], k=1)
                if results:
                    item = results[0]
            if item:
                items.append(item)
        if len(items) < 2:
            return "I could not find one or both of the assessments mentioned. Please provide exact names.", [], False

        # Build comparison prompt
        details = []
        for i in items:
            details.append(
                f"Name: {i['title']}\n"
                f"URL: {i['url']}\n"
                f"Test Types: {', '.join(i['test_types'])}\n"
                f"Job Levels: {', '.join(i['job_levels'])}\n"
                f"Duration: {i['assessment_length_minutes']} min\n"
                f"Remote Testing: {i['remote_testing']}\n"
                f"IRT: {i['adaptive_irt']}\n"
                f"Description: {i['description']}"
            )
        detail_text = "\n\n".join(details)
        prompt = (
            f"Compare these two SHL assessments based ONLY on the provided data:\n{detail_text}\n\n"
            "Highlight differences in test type, duration, job levels, content. "
            "Format your answer as a Markdown table if possible, or a clear factual comparison."
        )
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You compare assessments factually."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )
            reply = resp.choices[0].message.content.strip()
            return reply, [], True
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            return "Sorry, I couldn't generate the comparison at this moment.", [], False

    async def _generate_clarification_async(self, messages: List[Message]) -> str:
        prompt = f"Conversation:\n{self._format_history(messages)}\n\nThe request is too vague. Ask ONE clarifying question."
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You help clarify hiring needs."},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=80
            )
            return resp.choices[0].message.content.strip()
        except:
            return "Could you tell me about the role you're hiring for (e.g., job title, seniority, skills)?"

    async def _generate_reply_async(self, messages: List[Message], constraints: Dict, candidates: List[Dict]) -> Tuple[str, str]:
        candidate_str = "\n".join([
            f"{i+1}. {c['title']} ({', '.join(c['test_types'])}) - {c['description'][:120]}..."
            for i, c in enumerate(candidates[:10])
        ])
        user_prompt = USER_PROMPT_TEMPLATE.format(
            history=self._format_history(messages),
            candidates=candidate_str if candidates else "None"
        )
        full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": full_prompt}],
                temperature=0.4,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            reply = data.get("reply", "")
            action = data.get("action", "recommend")
            # Sanity check: if action recommend but no candidates, force clarify
            if action == "recommend" and not candidates:
                action = "clarify"
                reply = "I need a bit more information to make a recommendation."
            # Validate no hallucinated URLs in reply
            reply = self._sanitize_urls(reply)
            return reply, action
        except Exception as e:
            logger.error(f"Reply generation error: {e}")
            if candidates:
                rec_names = ", ".join([c['title'] for c in candidates[:5]])
                return f"Here are some assessments that might fit: {rec_names}.", "recommend"
            return "I'm having trouble generating a response. Could you rephrase?", "clarify"

    def _sanitize_urls(self, text: str) -> str:
        """Remove any URLs not in the catalog."""
        urls = re.findall(r'https?://[^\s]+', text)
        for u in urls:
            if not self.retriever.get_by_url(u):
                text = text.replace(u, "[assessment link]")
        return text

    def _format_history(self, messages: List[Message]) -> str:
        return "\n".join([f"{m.role}: {m.content}" for m in messages])


def create_agent(retriever: HybridRetriever) -> Agent:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    model = os.getenv("MODEL_NAME", "deepseek-chat")
    return Agent(retriever, client, model)