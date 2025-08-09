// Local: app/api/chat/route.ts

import { OpenAIEmbeddings } from "@langchain/openai";
import { UpstashVectorStore } from "@langchain/community/vectorstores/upstash";
import { Index } from "@upstash/vector";
import type { Document } from "@langchain/core/documents";
import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";

// Executa a API na Edge Network da Vercel para baixa latência
export const runtime = 'edge';

// Função helper para formatar documentos como string
const formatDocumentsAsString = (docs: Document[]): string => {
  return docs.map((doc) => doc.pageContent).join("\n\n");
};

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    const question = messages[messages.length - 1].content;

    // 1. Inicializa os clientes
    const index = new Index({
      url: process.env.UPSTASH_VECTOR_REST_URL!,
      token: process.env.UPSTASH_VECTOR_REST_TOKEN!,
    });
    const embeddings = new OpenAIEmbeddings({ modelName: "text-embedding-3-small" });

    // 2. Inicializa o Vector Store e busca documentos relevantes
    const vectorStore = new UpstashVectorStore(embeddings, { index });
    const retriever = vectorStore.asRetriever(4);
    const relevantDocs = await retriever.invoke(question);
    const context = formatDocumentsAsString(relevantDocs);

    // 3. Usa streamText do AI SDK v5 para gerar resposta
    const result = streamText({
      model: openai('gpt-4o-mini'),
      temperature: 0.2,
      system: `Você é um assistente especialista em teoria musical chamado Sebastian. Sua missão é responder à pergunta do usuário de forma clara e precisa, baseando-se SOMENTE no contexto fornecido abaixo.
Se a informação não estiver no contexto, diga "Com base no meu conhecimento atual, não encontrei informações sobre isso." Não invente respostas. Seja sempre cordial.

Contexto:
${context}`,
      prompt: question,
    });

    // 4. Retorna a resposta em streaming usando a nova API do AI SDK v5
    return result.toTextStreamResponse();

  } catch (e) {
    if (e instanceof Error) {
        console.error(e);
        return new Response(JSON.stringify({ error: e.message }), { status: 500 });
    }
    return new Response(JSON.stringify({ error: 'An unknown error occurred' }), { status: 500 });
  }
}
