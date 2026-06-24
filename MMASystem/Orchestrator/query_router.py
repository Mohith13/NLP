class QueryRouter:
    def route(self, query):
        q = query.lower()

        summarizer_keywords = [
            "summarize",
            "summary",
            "overview",
            "brief",
            "short notes",
            "explain in short"
        ]

        finder_keywords = [
            "find",
            "search",
            "extract",
            "show",
            "show text",
            "where is",
            "reference",
            "source"
        ]

        if any(keyword in q for keyword in summarizer_keywords):
            return "summarizer"

        if any(keyword in q for keyword in finder_keywords):
            return "text_finder"

        return "qa"