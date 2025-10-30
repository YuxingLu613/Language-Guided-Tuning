import re

def clean_leading_stars(text):
    """Remove leading lines that are empty or contain only asterisks."""
    if not text:
        return text
    lines = text.splitlines()
    start_idx = 0
    for line in lines:
        stripped = line.strip()
        if stripped == '' or all(c == '*' for c in stripped):
            start_idx += 1
        else:
            break
    cleaned = '\n'.join(lines[start_idx:]).strip()
    return cleaned

def parse_judger_suggestions(judger_output):
    """
    Parse the judger's output to extract suggestions for each agent.
    Handles both:
      - AUGMENTATION_AGENT_SUGGESTION:
      - **AUGMENTATION_AGENT_SUGGESTION:**
    and removes any leading asterisk-only lines from the content.
    """
    if not judger_output:
        return {}
    
    judger_output_str = str(judger_output) if judger_output is not None else ""
    agent_guidance = {}
    
    def extract_section(keyword):
        # Match optional ** before and after the keyword
        pattern = rf'(?:\*\*)?{re.escape(keyword)}(?:\*\*)?:\s*(.*?)(?=\n(?:\*\*)?[A-Z_]+(?:\*\*)?:|$)'
        match = re.search(pattern, judger_output_str, re.DOTALL)
        if match:
            content = match.group(1)
            return clean_leading_stars(content)
        return None

    aug = extract_section("AUGMENTATION_AGENT_SUGGESTION")
    if aug:
        agent_guidance['aug'] = aug

    adaptive = extract_section("ADAPTIVE_AGENT_SUGGESTION")
    if adaptive:
        agent_guidance['adaptive'] = adaptive

    hpo = extract_section("HPO_AGENT_SUGGESTION")
    if hpo:
        agent_guidance['hpo'] = hpo

    return agent_guidance


if __name__ == "__main__":
    # 测试：模拟你遇到的带多余 ** 的情况
    judger_output = """Some intro...

**AUGMENTATION_AGENT_SUGGESTION:**
**
- Focus on data normalization.

ADAPTIVE_AGENT_SUGGESTION:
- Continue using Huber loss.

**HPO_AGENT_SUGGESTION:**
**
- Remove dropout.
"""

    result = parse_judger_suggestions(judger_output)
    print(result)