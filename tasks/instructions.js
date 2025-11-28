import {N_BLOCKS, N_POINTS_PER_POUND, MAX_BONUS_POUNDS} from './taskParams.js';

import {version, VERSION_CHANGE_POINT, VERSION_RANDOM_WALK} from './version.js';

const N_INSTRUCTIONS_CP = 14;
const QUIZ_INSTRUCTION_IDX_CP = 11;
const INSTRUCTIONS_PATH_CP = "instructions-cp.html";
const N_INSTRUCTIONS_RW = 15;
const QUIZ_INSTRUCTION_IDX_RW = 12;
const INSTRUCTIONS_PATH_RW = "instructions-rw.html";

export const n_instructions = (version === VERSION_RANDOM_WALK ?
  N_INSTRUCTIONS_RW : N_INSTRUCTIONS_CP);
const quiz_instruction_idx = (version === VERSION_RANDOM_WALK ?
  QUIZ_INSTRUCTION_IDX_RW : QUIZ_INSTRUCTION_IDX_CP);
const instructions_path = (version === VERSION_RANDOM_WALK ?
  INSTRUCTIONS_PATH_RW : INSTRUCTIONS_PATH_CP);
export let currInstructionIdx = 0;
export let hasSucceededQuiz = false;
export let didJustFailQuiz = false;

let _currInstructionWillChangeCallback = null;
let _currInstructionDidChangeCallback = null;
let _quizDidSubmitAnswersCallback = null

export async function loadInstructions(onloadCallback,
  startHandler,
  currInstructionWillChangeCallback,
  currInstructionDidChangeCallback,
  quizDidSubmitAnswersCallback,
  nBlocks = N_BLOCKS,
  showBonusText = false,
  containerId = "instruction-container") {
  // Fetch instructions content
  let response = await fetch(instructions_path);
  let instructions = await response.text();
  // Embed the instructions HTML within the instructions content container in the document tree
  const instructionContainer = document.getElementById(containerId);
  instructionContainer.innerHTML = instructions;
  // Finalize the html initial setup
  setupInstructions(startHandler, nBlocks, showBonusText);
  updateInstruction();
  // Run the callback once the instructions have been loaded
  if (onloadCallback) {
    onloadCallback();
  }
  _currInstructionWillChangeCallback = currInstructionWillChangeCallback;
  _currInstructionDidChangeCallback = currInstructionDidChangeCallback;
  _quizDidSubmitAnswersCallback = quizDidSubmitAnswersCallback;
  // Preload the images for the instructions.
  // At the moment, programmatically preloading the images does not seem
  // to be necessary as the preloading already happens when the HTML is set up.
  // However, this may change in the future so I am keeping the preloading code
  // for reference.
  // preloadInstructionImages();
}

export function setupInstructions(startHandler,
  nBlocks = N_BLOCKS, showBonusText = false) {
  const totalInstructions = document.getElementById("total-instructions"); 
  totalInstructions.textContent = n_instructions.toString();
  document.getElementById("previous-instruction-button").onclick = prevInstruction;
  document.getElementById("next-instruction-button").onclick = nextInstruction;
  if (startHandler) {
    document.getElementById("start-or-resume-task-button").onclick = startHandler;
  }
  document.getElementById("quiz-submit-button").onclick = checkQuizAnswers;
  document.getElementById("review-instructions-button").onclick = reviewInstructions;
  document.getElementById("n-blocks").textContent = nBlocks.toString();
  const bonusText = document.getElementById("bonus-text");
  bonusText.style.display = showBonusText ? "" : "none";
  if (showBonusText) {
    document.getElementById("points-per-pound").textContent = N_POINTS_PER_POUND.toString();
    document.getElementById("max-bonus-pounds").textContent = MAX_BONUS_POUNDS.toFixed(2).toString();
  }
}

export function prevInstruction() {
  if (currInstructionIdx > 0) {
    _currInstructionWillChangeCallback?.(currInstructionIdx, currInstructionIdx-1);
    currInstructionIdx--;
    updateInstruction();
    _currInstructionDidChangeCallback?.();
  }
}

export function nextInstruction() {
  if (currInstructionIdx < n_instructions - 1) {
    _currInstructionWillChangeCallback?.(currInstructionIdx, currInstructionIdx+1);
    currInstructionIdx++;
    updateInstruction();
    _currInstructionDidChangeCallback?.();
  }
}

export function goToInstructionIdx(idx) {
  if ((idx >= 0) && (idx < n_instructions)) {
    _currInstructionWillChangeCallback?.(currInstructionIdx, idx);
    currInstructionIdx = idx;
    updateInstruction();
    _currInstructionDidChangeCallback?.();
  }
}

export function reviewInstructions() {
  didJustFailQuiz = false;
  // clearQuizAnswers();
  _currInstructionWillChangeCallback?.(currInstructionIdx, 0);
  currInstructionIdx = 0;
  updateInstruction();
  _currInstructionDidChangeCallback?.();
}

export function updateInstruction() {
  // Clear quiz feedback element after changing the instruction
  clearQuizFeedback();
  // Update which of the instructions is displayed based on the current index
  for (let i = 0; i < n_instructions; i++) {
    const instruction = document.getElementById(`instruction-${i+1}`);
    instruction.style.display = (i == currInstructionIdx) ? "" : "none";
  }
  // Update the rank of the current instruction displayed in the navigation bar
  const currentInstructionRank = document.getElementById("current-instruction-rank");
  currentInstructionRank.textContent = (currInstructionIdx + 1).toString();
  // Update navigation buttons visibility and enabled state.
  updateNavButtons();
}

export function updateNavButtons() {
  // Disable/Enable or Hide/Show navigation buttons based on the current
  // instruction index and whether the user needs to perform a quiz before
  // proceeding, or has passed or failed the quiz.
  const isFirstInstruction = currInstructionIdx === 0;
  const isLastInstruction = currInstructionIdx === n_instructions - 1;
  const isQuizInstruction = currInstructionIdx == quiz_instruction_idx;
  const shouldPerformQuiz = isQuizInstruction && !hasSucceededQuiz;
  document.getElementById("previous-instruction-button").disabled = (isFirstInstruction
    || shouldPerformQuiz || didJustFailQuiz);
  document.getElementById("next-instruction-button").disabled = (isLastInstruction
      || shouldPerformQuiz || didJustFailQuiz);
  document.getElementById("start-or-resume-task-button").style.visibility = (
      (isLastInstruction) ? "visible" : "hidden");
  // replace next/previous buttons by the review instructions button when the user just failed the quiz
  document.getElementById("instruction-nav-next-prev-container").style.display = (
      (isQuizInstruction && didJustFailQuiz) ? "none" : "")
  document.getElementById("review-instructions-button").style.display = (
      (isQuizInstruction && didJustFailQuiz) ? "" : "none");
  document.getElementById("quiz-submit-button").disabled = (
      (isQuizInstruction && didJustFailQuiz));
}

export function checkQuizAnswers() {
  // Correct answers
  const correctAnswers = (version === VERSION_RANDOM_WALK ? {
      q1: 'D',
      q2: 'D',
      q3: 'A',
      q4: 'C',
    } : {
      q1: 'C',
      q2: 'D',
      q3: 'A',
      q4: 'B',
      q5: 'C'
  });

  // User answers
  let allCorrect = true;
  let feedbackMessage = '';
  let answers = [];

  for (let question in correctAnswers) {
      const selectedOption = document.querySelector(
        `input[name="${question}"]:checked`);
      if (selectedOption) {
          const answer = selectedOption.value;
          if (answer !== correctAnswers[question]) {
              allCorrect = false;
          }
          answers.push(answer);
      } else {
          allCorrect = false;
      }
  }

  handleQuizResult(allCorrect);

  _quizDidSubmitAnswersCallback?.(answers);
}

export function handleQuizResult(allCorrect) {
  if (allCorrect) {
    hasSucceededQuiz = true;
  } else {
    didJustFailQuiz = true;
  }
  // Provide feedback
  updateQuizFeedback(allCorrect);
  // Update buttons to either allow the participant to proceed or prompt them to
  // review the instructions
  updateNavButtons();
}

export function updateQuizFeedback(allCorrect) {
  const feedbackElement = document.getElementById("quiz-feedback");
  if (allCorrect) {
    feedbackElement.innerHTML = "All answers are correct! You can proceed.";
  } else {
    feedbackElement.innerHTML = "At least one of your answers was incorrect.\
      Please review the instructions and try again.";
  }
}

export function clearQuizFeedback() {
  const feedbackElement = document.getElementById("quiz-feedback");
  feedbackElement.innerHTML = "";
}

export function clearQuizAnswers() {
  for (let question of ['q1', 'q2', 'q3', 'q4', 'q5']) {
    const options = document.querySelectorAll(
      `input[name="${question}"]`);
    for (let option of options) {
      option.checked = false;
    }
  }
}

function preloadInstructionImages() {
  // Preload all images that are going to be displayed in the instructions
  const imgElements = document.getElementById(
    "instruction-content-container").getElementsByTagName("img");
  // Iterate over instruction images
  for (let imgElement of imgElements) {
    // There are multiple image sources depending on the device pixel density.
    // We want to preload the image source corresponding the current density.
    const srcset = imgElement.getAttribute("srcset");
    const dpr = window.devicePixelRatio || 1;
    // Split srcset into image candidates (e.g., "low-res.png 1x", "high-res.png 2x")
    const imageCandidates = srcset.split(',').map(item => {
      const [src, resolution] = item.trim().split(' ');
      return { src: src, resolution: parseFloat(resolution) || 1 };
    });
    // Find the image with the closest resolution to the devicePixelRatio
    let selectedImage = imageCandidates[0];
    for (const candidate of imageCandidates) {
      if (candidate.resolution >= dpr) {
        selectedImage = candidate;
        break;
      }
    }
    // Preload the selected image
    preloadImage(selectedImage.src);
    console.log(`preload image: ${selectedImage.src} (res=${selectedImage.resolution})`);
  }
}

function preloadImage(src) {
  var img = new Image();
  img.src = src;  
}
