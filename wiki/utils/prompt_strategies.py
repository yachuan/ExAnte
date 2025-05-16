class PromptStrategies:
    @staticmethod
    def zero_shot(title, cutoff_year):
        return f"Regarding your knowledge about '{title}', generate 5 atomic facts. Each fact should be a short, clear statement representing a distinct piece of information. Only use information from before January 1st, {cutoff_year}."

    @staticmethod
    def instruction_based(title, cutoff_year):
        return f"Regarding your knowledge about '{title}', generate 5 atomic facts. Each fact should be a short, clear statement representing a distinct piece of information. Only use information that was available before January 1st, {cutoff_year}."

    @staticmethod
    def chain_of_thought(title, cutoff_year):
        return f"Regarding your knowledge about '{title}', let's think step by step to generate 5 atomic facts. Each fact should be a short, clear statement representing a distinct piece of information. Only consider information available before January 1st, {cutoff_year}."

    @staticmethod
    def few_shot(title, cutoff_year):
        return f"""Here are some examples of good atomic facts about different topics:

Topic: World War II cutoff: 1946
- The war began in Europe with Germany's invasion of Poland in September 1939 (valid because it's before the cutoff)
- The United States entered the war after the Pearl Harbor attack in December 1941 (valid because it's before the cutoff)

Topic: World War II cutoff: 1941
- The war ended in Europe on May 7, 1945, with Germany's surrender (invalid because it's after the cutoff)

Now, regarding '{title}', generate 5 atomic facts. Each fact should be a short, clear statement representing a distinct piece of information. Only use information from before January 1st, {cutoff_year}."""

    @staticmethod
    def extract_facts_from_generate_validate(response):
        """Helper method to extract just the facts from generate_and_validate response"""
        facts = []
        for line in response.split('\n'):
            if line.strip().startswith('[Fact]:'):
                facts.append(line.split('[Fact]:')[1].strip())
        return facts 