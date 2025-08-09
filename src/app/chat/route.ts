// Local: app/api/chat/route.ts

import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { UpstashVectorStore } from "@langchain/community/vectorstores/upstash";
import { Index } from "@upstash/vector";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
// CORREÇÃO: Importações da biblioteca 'ai' atualizadas
import { LangChainAdapter, StreamingTextResponse } from "ai";

// Executa a API na Edge Network da Vercel para baixa latência
export const runtime = 'edge';

// O modelo do prompt que instrui a IA sobre como se comportar.
const template = `
Você é um assistente especialista em teoria musical chamado Sebastian. Sua missão é responder à pergunta do usuário de forma clara e precisa, baseando-se SOMENTE no contexto fornecido abaixo.
Se a informação não estiver no contexto, diga "Com base no meu conhecimento atual, não encontrei informações sobre isso." Não invente respostas. Seja sempre cordial.

Contexto:
{context}

Pergunta:
{question}

Resposta útil:
`;

const prompt = PromptTemplate.fromTemplate(template);

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
    const llm = new ChatOpenAI({
      modelName: "gpt-4o-mini",
      temperature: 0.2,
      streaming: true,
    });

    // 2. Inicializa o Vector Store e o Retriever
    const vectorStore = new UpstashVectorStore({ index, embeddings });
    const retriever = vectorStore.asRetriever(4); // Busca os 4 chunks mais relevantes

    // 3. Cria a "Chain" (cadeia de execução) com LangChain
    const chain = RunnableSequence.from([
      {
        context: retriever.pipe(formatDocumentsAsString),
        question: new RunnablePassthrough(),
      },
      prompt,
      llm,
      new StringOutputParser(),
    ]);

    // 4. CORREÇÃO: Usa o LangChainAdapter para criar um stream compatível
    const stream = await chain.stream(question);

    // 5. Retorna a resposta em streaming para o cliente
    return new StreamingTextResponse(stream);

  } catch (e) { // CORREÇÃO: Removido o tipo 'any' para satisfazer o ESLint
    if (e instanceof Error) {
        console.error(e);
        return new Response(JSON.stringify({ error: e.message }), { status: 500 });
    }
    return new Response(JSON.stringify({ error: 'An unknown error occurred' }), { status: 500 });
  }
}
