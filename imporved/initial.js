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

// Job Description as Context
const jobDescription = `
Job Title: Junior Software Engineer
Company: TechCorp
Description: TechCorp is seeking a Junior Software Engineer to join our innovative team. The candidate will work on developing web applications using JavaScript, React, and Node.js. Responsibilities include writing clean code, collaborating with cross-functional teams, and participating in agile development processes. The ideal candidate is passionate about technology, has strong problem-solving skills, and is eager to learn. Required skills: JavaScript, React, Node.js, teamwork, and communication. Preferred: Experience with cloud platforms (e.g., AWS).
Company Values: Innovation, Collaboration, Continuous Learning.
`;

// Create Documents from Job Description
const docs = [new Document({ pageContent: jobDescription })];

// Text Splitter
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});

// Instantiate Embeddings
const embeddings = new OpenAIEmbeddings();

// Create Vector Store
async function initializeVectorStore() {
  const splitDocs = await splitter.splitDocuments(docs);
  return await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
}

// Create Retriever and Chains
async function initializeChains() {
  const vectorstore = await initializeVectorStore();
  const retriever = vectorstore.asRetriever({ k: 2 });

  // History-Aware Retriever Prompt
  const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up information relevant to the conversation and the job description.",
    ],
  ]);

  // Create History-Aware Retriever
  const retrieverChain = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retrieverPrompt,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an HR interviewer trainee that helps students practice for interviews based on a job description at TechCorp. The position is: {context}.
  
  Your behavior:
  - Begin by greeting the candidate, introducing yourself as the HR representative from TechCorp, and ask the candidate to introduce themselves.
  - Do not specify your own name or use placeholders for the candidate’s name.
  - Ask questions based on the job description or based on their personal introduction (e.g., skills, tools, or past experience).
  - Avoid saying “moving on to a different topic from the job description.” 
    - Instead, say “Let’s go to the next question,” “Next up,” or just proceed to the next question naturally.
  
  Questioning behavior:
  - Do **not** start every question with “Thank you for responding.”
  - Maintain a professional, friendly, and natural tone.
  - Ask 1-2 follow-up questions per topic.
    - If the answer is strong, ask a deeper or more technical follow-up.
    - If it’s weak or unclear, rephrase or simplify it.
    - After 1-2 follow-ups, change to a different relevant topic based on the job description.
  - If the candidate gives similar or generic answers across multiple questions:
    - Gently point this out by saying it might not fit the current context.
    - Ask for more specific examples or a more tailored answer.
    - Provide suggestions to improve the response and offer them a chance to revise or move on.
  
 
  
  Answer guidance:
  - If the candidate asks: “How should I answer this?” or “Can you give me a hint?” — provide general guidance:
    - “Share a specific experience or situation,”
    - “Mention tools or technologies used,”
    - “Use a structure: context, action, result,”
    - “Connect your answer to a real-world project.”
  
     Feedback after each response:
  1. Acknowledge the response naturally (e.g., “Got it”, “Okay”, “Makes sense”).
  2. If the answer was too short, vague, or lacking depth—especially in the introduction (e.g., just saying "I'm XYZ"):
     - Encourage the candidate to expand by mentioning things like their education, skills, projects, interests, or career goals.
     - Example guidance: “In interviews, it's helpful to include a bit more detail about your background, such as your degree, what you're passionate about, and any projects or tools you've worked with.”
     - After each feedback ask the user do they want to answer again or will move towards next question.
  3. Highlight what was good and suggest what could be added or improved.
  4. Ask: “Would you like to revise your answer, or should we move to the next question?”
  
    Inconsistency detection:
  - Actively remember the candidate’s previous answers during the session.
  - If the candidate mentions contradictory or mismatching details (e.g., different technologies used for the same project), politely point it out.
  - Example: If the candidate first says they used React Native and later says they used Next.js for the same project, ask for clarification.
  - Example prompt:
    “Earlier, you mentioned you used React Native for this project, but now you mentioned Next.js. Could you clarify which one you used, or were both involved in different ways?”

  Closing or exit behavior:
  - If the candidate says something like “Let’s end this,” “That’s enough,” or “I want to stop,” or you think the quetion is enough.
  - Confirm their intent: “Would you like to continue the interview or end it here?”
  - If they choose to continue, resume with the next question naturally.
  - If they choose to end, politely thank them and provide the following.
    - General feedback on their performance (e.g., strengths and areas for improvement).
    - A table summarizing:
      | Question | User's Answer (Short Summary) | Feedback |
    - note*- the question and use answers in tabular format should be one words "like the introduction, tech stack  or the exprerience "
  `,
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);

  // Create Document Chain
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });

  // Create Conversation Chain
  return await createRetrievalChain({
    combineDocsChain: chain,
    retriever: retrieverChain,
  });
}

// Conduct Interview in Terminal
async function conductInterview() {
  const conversationChain = await initializeChains();
  let chatHistory = [];

  console.log("=== TechCorp Junior Software Engineer Interview ===");
  console.log("Type 'exit' at any time to end the interview.\n");

  // Start with an initial question
  let response = await conversationChain.invoke({
    chat_history: chatHistory,
    input: "Start the interview with an initial question.",
  });

  console.log("HR:", response.answer);
  chatHistory.push(new AIMessage(response.answer));

  // Main interview loop
  while (true) {
    const studentAnswer = await new Promise((resolve) =>
      rl.question("Your Answer: ", resolve)
    );

    if (studentAnswer.toLowerCase() === "exit") {
      console.log("=== Interview Ended ===");
      break;
    }

    // Update chat history with student's answer
    chatHistory.push(new HumanMessage(studentAnswer));

    // Generate follow-up question
    response = await conversationChain.invoke({
      input: studentAnswer,
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
