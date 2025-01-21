class PromptDesigner:
    def __init__(self):
        self.persona_task_description = "..."
        self.domain_localization = "..."
        # Add all other prompt components

    def generate_prompt(self, **kwargs):
        prompt = ""
        if kwargs.get("include_persona"):
            prompt += self.persona_task_description + "\n"
        if kwargs.get("include_domain"):
            prompt += self.domain_localization + "\n"
        # Append other parts conditionally
        return prompt
