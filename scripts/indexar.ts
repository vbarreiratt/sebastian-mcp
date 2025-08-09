// Local: scripts/indexar.ts

// Para executar este script, você pode precisar do tsx.
// Instale com: pnpm add -D tsx
// Execute com: pnpm tsx scripts/indexar.ts

import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { UpstashVectorStore } from "@langchain/community/vectorstores/upstash";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { Index } from "@upstash/vector";
import * as dotenv from "dotenv";
import * as path from "path";

// CORREÇÃO: Carrega explicitamente as variáveis do arquivo .env.local
// Isso garante que o script encontre as chaves mesmo rodando fora do ambiente Next.js
dotenv.config({ path: path.resolve(process.cwd(), ".env.local") });


// 1. Validação das Chaves de API
// Verifica se as chaves necessárias estão presentes no ambiente.
if (
  !process.env.UPSTASH_VECTOR_REST_URL ||
  !process.env.UPSTASH_VECTOR_REST_TOKEN ||
  !process.env.OPENAI_API_KEY
) {
  console.error(
    "Erro: As variáveis de ambiente UPSTASH_VECTOR_REST_URL, UPSTASH_VECTOR_REST_TOKEN e OPENAI_API_KEY precisam estar definidas no seu arquivo .env.local"
  );
  process.exit(1);
}

// 2. Configuração dos Clientes
const index = new Index({
  url: process.env.UPSTASH_VECTOR_REST_URL,
  token: process.env.UPSTASH_VECTOR_REST_TOKEN,
});

const embeddings = new OpenAIEmbeddings({
  modelName: "text-embedding-3-small",
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// 3. Carregamento dos Documentos da Pasta
const loader = new DirectoryLoader(
  './dados_musicais', // A pasta que criamos
  {
    '.txt': (path) => new TextLoader(path),
  }
);

async function main() {
  try {
    console.log("Carregando documentos da pasta 'dados_musicais'...");
    const docs = await loader.load();

    if (docs.length === 0) {
        console.log("Nenhum documento .txt encontrado na pasta. Encerrando.");
        return;
    }

    // 4. Quebra dos Documentos (Chunking)
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 100,
    });
    const splitDocs = await splitter.splitDocuments(docs);
    console.log(`Documentos divididos em ${splitDocs.length} pedaços.`);

    // 5. Indexação no Upstash Vector
    console.log("Iniciando a indexação no Upstash Vector... Isso pode levar alguns segundos.");
    await UpstashVectorStore.fromDocuments(splitDocs, embeddings, {
      index,
    });
    console.log("✅ Indexação concluída com sucesso!");
    console.log("Seu banco de dados vetorial agora contém o conhecimento dos arquivos.");

  } catch (error) {
    console.error("❌ Ocorreu um erro durante a indexação:", error);
  }
}

main();
