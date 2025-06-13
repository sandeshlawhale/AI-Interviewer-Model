import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";
import readline from "readline";
import { stdin as input, stdout as output } from "process";

// Import environment variables
import * as dotenv from "dotenv";
dotenv.config();

// Create readline interface
const rl = readline.createInterface({ input, output });

// Instantiate Model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

const jobDescription = `
Job Title: Junior Software Engineer
Company: TechCorp
Description: TechCorp is seeking a Junior Software Engineer to join our innovative team. The candidate will work on developing web applications using JavaScript, React, and Node.js. Responsibilities include writing clean code, collaborating with cross-functional teams, and participating in agile development processes. The ideal candidate is passionate about technology, has strong problem-solving skills, and is eager to learn. Required skills: JavaScript, React, Node.js, teamwork, and communication. Preferred: Experience with cloud platforms (e.g., AWS).
Company Values: Innovation, Collaboration, Continuous Learning.
`;
const mainPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a HR interviewer of company mentioned in the JD conducting a mock interview and the JD is: {context}.
  
  Your role:
  - Begin with a friendly greeting and introducing yourself, for your name choose any from the indian male names.
  - Ask them to introduce themselves.
  - Speak in a natural, casual tone — not overly formal or scripted.
  - Avoid robotic phrases like “I’m excited to learn more...” or “Hello and welcome...” — keep it real and human.
  - Use smooth transitions like “Let’s go to the next question” or move to new topics naturally.
  - Ask 1–2 follow-ups per topic.
    - Go deeper if the answer is strong.
    - Rephrase/simplify if it's unclear.
  - If answers are generic or repeated:
    - Point it out gently
    - Ask for specifics
    - Suggest improvements and offer a chance to retry.
  - If they ask for help, offer tips:
    - “Mention tools or examples”
    - “Use the STAR format”
    - “Tie it to a real project or goal”
  - Remember past answers and clarify inconsistencies.`,
  ],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
]);
const feedbackPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are an HR assistant giving feedback after each answer.
  
  - Acknowledge naturally (e.g., “Got it”, “Okay”).
  - If vague or too short:
    - Suggest adding education, skills, tools, projects, or goals.
    - Ex: “as asked in question, you can Mention your background, passions, and tools you've used or few real world examples if any to improve your response.”
  - If the answer might have the revise condition from the feedback then End with: “Would you like to revise or move to the next question?”`,
  ],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
]);
const finalFeedbackPrompt = ChatPromptTemplate.fromMessages([
  new MessagesPlaceholder("chat_history"),
  [
    "system",
    `Summarize the interview by returning a JavaScript array of objects.
  Each object should include:
  
  - question: question asked by interviewer avoid any personal touch here.
  - response: The user's original answer that user provided to that answer, pick this answer form chat history, it should be as it is and it is human response
  - feedback: Specific, constructive feedback
  - strengths: Key strengths shown in this answer
  - improvements: Specific areas to improve in this answer
  - response_depth: One of ["Level 1", "Level 2", "Level 3"]
  
  Output only a valid JavaScript array. Be concise, objective, and helpful.`,
  ],
]);

//load docs and create vector store
async function initializeVectorStore() {
  const docs = [new Document({ pageContent: jobDescription })];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1500,
    chunkOverlap: 20,
  });

  const embeddings = new OpenAIEmbeddings();

  const splitDocs = await splitter.splitDocuments(docs);
  return await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
}

// Create Retriever and Chains
async function initializeChains() {
  const vectorstore = await initializeVectorStore();
  const retriever = vectorstore.asRetriever({ k: 2 });

  const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up information relevant to the conversation and the job description.",
    ],
  ]);

  const retrieverChain = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retrieverPrompt,
  });

  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: mainPrompt,
  });

  return await createRetrievalChain({
    combineDocsChain: chain,
    retriever: retrieverChain,
  });
}

async function conductInterview() {
  const conversationChain = await initializeChains();

  const finalFeedbackChain = finalFeedbackPrompt.pipe(model);
  const feedbackChain = feedbackPrompt.pipe(model);

  let chatHistory = [];

  console.log("=== TechCorp Junior Software Engineer Interview ===");
  console.log("Type 'exit' at any time to end the interview.\n");

  // Start with an initial question
  let response = await conversationChain.invoke({
    input: "Start the interview with an initial question.",
    chat_history: chatHistory,
  });

  console.log("HR:", response.answer);
  chatHistory.push(new AIMessage(response.answer));

  // Main interview loop
  while (true) {
    const studentAnswer = await new Promise((resolve) =>
      rl.question("Your Answer: ", resolve)
    );

    const lower = studentAnswer.toLowerCase();
    if (lower === "exit") {
      console.log(
        "\nHR: Would you like to continue the interview or end it here?"
      );
      const confirmation = await new Promise((resolve) =>
        rl.question("Your Answer (yes to continue / no to end): ", resolve)
      );
      if (confirmation.toLowerCase().startsWith("n")) {
        console.log(
          "\nHR: Thank you for your time! Here's your overall feedback:\n"
        );

        const finalFeedback = await finalFeedbackChain.invoke({
          chat_history: chatHistory,
        });

        console.log(finalFeedback.content);
        console.log("chatHistory", chatHistory);
        chatHistory = [];
        break;
      } else {
        console.log("HR: Great! Let's continue.\n");
        continue;
      }
    }

    chatHistory.push(new HumanMessage(studentAnswer));

    // 💬 Step 1: Give Feedback
    const feedbackResponse = await feedbackChain.invoke({
      input: studentAnswer,
      chat_history: chatHistory,
    });

    console.log("\nHR (Feedback):", feedbackResponse.content);
    // chatHistory.push(new AIMessage(feedbackResponse.content));

    // Ask if they want to revise
    const reviseChoice = await new Promise((resolve) =>
      rl.question(
        "Would you like to revise your answer or want to move on (yes to revise): ",
        resolve
      )
    );

    if (reviseChoice.toLowerCase().includes("yes")) {
      const revisedAnswer = await new Promise((resolve) =>
        rl.question("Your Revised Answer: ", resolve)
      );
      chatHistory.pop();
      chatHistory.push(new HumanMessage(revisedAnswer));
    }

    // 💬 Step 2: Next Question
    response = await conversationChain.invoke({
      input:
        "Ask the next question or ask the follow up question to the previous one.",
      chat_history: chatHistory,
    });

    console.log("\nHR:", response.answer);
    chatHistory.push(new AIMessage(response.answer));
  }

  rl.close();
}

// Run the interview
conductInterview().catch((error) => {
  console.error("Error:", error);
  rl.close();
});
