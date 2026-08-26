import 'dotenv/config';
import { BaseDocumentCompressor } from '@langchain/core/retrievers/document_compressors'

export class DashScopeRerank extends BaseDocumentCompressor {
  constructor({ apiKey, model = 'qwen3-rerank', topN = 3, baseUrl } = {}) {
    super();
    this.apiKey = apiKey;
    this.model = model;
    this.topN = topN;
    this.baseUrl = baseUrl || process.env.RERANK_URL;
  }
  async compressDocuments(documents, query, _callbacks) {
    const res = await fetch(this.baseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        model: this.model,
        input: {
          query: query,
          documents: documents.map(doc => doc.pageContent)
        },
        parameters: {
          return_documents: false,
          top_n: this.topN
        }
      })
    });
    const json = await res.json();
    if (!res.ok) {
      throw new Error(`Rerank API request failed: ${json.error || res.statusText}`);
    }
    const results = json?.output?.results
    if (!Array.isArray(results)) {
      throw new Error(`Rerank API response format is unexpected: ${JSON.stringify(json)}`);
    }
    return results.map((item) => documents[item.index])
  }
}