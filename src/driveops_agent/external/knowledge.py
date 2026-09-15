from .base import Adapter


class KnowledgeAdapter(Adapter):
    def search_docs(self, query):
        return self.request("search_docs", {"query": query})

    def query_dtcs(self, code):
        return self.request("query_dtcs", {"code": code})
