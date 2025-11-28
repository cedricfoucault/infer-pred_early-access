import {requirementsCheckResults} from './check-requirements.js';

if (!requirementsCheckResults.passed) {
  throw new Error("Requirements not met");
}

import {TWOPI, DEG_TO_RAD, RAD_TO_DEG, mod, subtractAngle,
  range, randomSample} from './utils/num.js';

import {BET_WIDTHS_RAD, SCORE_GAIN_TABLE, INITIAL_VALUES,
  N_POINTS_PER_COIN, nCoinsWithPoints, N_BLOCKS, N_POINTS_PER_POUND,
  MAX_BONUS_POUNDS} from './taskParams.js';

import {parseURLParams} from './urlParams.js';

import {TaskDisplay, getCanvas, getTextBox, getHelpPanel, getPracticeHelpPanel,
  getInstructionContainer, getConsentContainer, getFeedbackForm} from './userInterface.js';

import {MouseTracker, MouseTarget} from './userInput.js';

import {SubjectData, BlockData, BlockEventData, ExperimentEventData} from './subjectData.js';

import {n_instructions, prevInstruction, nextInstruction, goToInstructionIdx,
  loadInstructions, currInstructionIdx} from './instructions.js';

import {loadConsent} from './consent.js';

import {PavloviaManager} from './pavlovia-manager.js';

const PROLIFIC_COMPLETION_URL = "https://app.prolific.com/submissions/complete?cc=CBJRQSZW";

// Delay used between saving two data files. For an unknown reason, uploading
// files to the Pavlovia server sometimes bugs when the several files are sent
// simultaneously. Adding a delay between two uploads helps resolve this issue.
const DATA_SAVE_DELAY = null; // ms.

let urlParams = parseURLParams();

import {version, VERSION_CHANGE_POINT, VERSION_RANDOM_WALK} from './version.js';
console.log("version", version);

const SEQ_DATA_PATH_CP = "./seq-data/seq-data_config-change-point-env_nseq-100_seed-1.json";
const SEQ_DATA_PATH_RW = "./seq-data/seq-data_config-random-walk-env_nseq-100_seed-1.json";
const seq_data_path = ((version === VERSION_RANDOM_WALK) ? SEQ_DATA_PATH_RW : SEQ_DATA_PATH_CP);

const subjectId = urlParams.subjectId ?? 'test';
const nBlocks = urlParams.nBlocks ?? N_BLOCKS;
let seqData;
let seqIndex;
let seqIndexPerBlock;
let expEventData = new ExperimentEventData(subjectId);
let subjectData = new SubjectData(subjectId);
let currBlockData;
let currBlockEventData;
let currBlockIdx;
let currTrialIndex;
let totalTaskScore = 0;
let isReady = false;

const TaskState = Object.freeze({
  LOADING: "loading",
  CONSENT: "consent",
  INSTRUCTIONS: "instructions",
  PRACTICE_MOVE_LOCATION_1: "practice_move_location_1",
  PRACTICE_MOVE_LOCATION_2: "practice_move_location_2",
  PRACTICE_SWITCH_LARGE_WID: "practice_switch_large_wid",
  PRACTICE_SWITCH_SMALL_WID: "practice_switch_small_wid",
  BLOCK: "block",
  BETWEEN_BLOCKS: "between_blocks",
  FINISHED: "finished"
});

let currentState = TaskState.LOADING;

function stateIsPractice(state) {
  return ((state == TaskState.PRACTICE_MOVE_LOCATION_1)
    || (state == TaskState.PRACTICE_MOVE_LOCATION_2)
    || (state == TaskState.PRACTICE_SWITCH_LARGE_WID)
    || (state == TaskState.PRACTICE_SWITCH_SMALL_WID));
}

const PRACTICE_MOVE_LOCATION_PRECEDING_INSTRUCTION_IDX = (version == VERSION_RANDOM_WALK ?
  5 : 5);
const PRACTICE_SWITCH_WID_PRECEDING_INSTRUCTION_IDX = (version == VERSION_RANDOM_WALK ?
  6 : 6);
const PRACTICE_TARGET_LOCATION_1 = 150 * DEG_TO_RAD;
const PRACTICE_TARGET_LOCATION_2 = (version == VERSION_RANDOM_WALK ?
  120 * DEG_TO_RAD : 30 * DEG_TO_RAD);
const PRACTICE_DELAY = 500; // ms

export const PRACTICE_TEXT = {
  TARGET_LOCATION: ("Practice moving the paddle using the mouse or keyboard " +
      "controls provided below.\nTo complete this practice, bring the paddle to " +
      "the target location (black dot)."),
  SWITCH_LARGE_WID: ("Practice switching from the small paddle to the large paddle " +
    "using the mouse or keyboard controls provided below."),
  SWITCH_SMALL_WID: ("Practice switching from the large paddle to the small paddle " +
    "using the mouse or keyboard controls provided below."),
}

//
// Functions for drawing and updating the display
//

let needsRedraw;
function setNeedsRedraw() {
  needsRedraw = true;
  // if (!taskSessionRunLoop?.willUpdateFrame) {
    // Execute drawIfNeeded on the next event cycle,
    // so that multiple calls to setNeedsRedraw within the current cycle
    // will result in only one execution draw() in the first execution of
    // drawIfNeeded() (the other calls being shortcuted due to needsRedraw being false).
    // This is more efficient than calling drawIfNeeded directly here,
    // which would result in multiple redundant draw() executions if setNeedsRedraw()
    // is called several times in one cycle.
    setTimeout(drawIfNeeded, 0);
  // }
}

// Make setNeedsRedraw globally accessible so that UI objects
// can call it when their state changes
window.setNeedsRedraw = setNeedsRedraw; 

function drawIfNeeded() {
  if (needsRedraw) {
    draw();
  }
  needsRedraw = false;
}

function draw() {
  let ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  taskDisplay.draw(ctx);
}


//
// Functions for user input event handling
//

function stateAllowsPaddleControl(state) {
  return (state == TaskState.BLOCK
    || state == TaskState.PRACTICE_MOVE_LOCATION_1
    || state == TaskState.PRACTICE_MOVE_LOCATION_2
    || state == TaskState.PRACTICE_SWITCH_LARGE_WID
    || state == TaskState.PRACTICE_SWITCH_SMALL_WID);
}

// Mouse

function updateForMouseMovement(mouseEvent) {
  if (!stateAllowsPaddleControl(currentState)) {
    return;
  }
  // https://stackoverflow.com/questions/4131252/how-to-convert-mouse-movements-to-rotation-of-an-element
  const originX = taskDisplay.ring.centerX;
  const originY = taskDisplay.ring.centerY;
  const radius = taskDisplay.ring.midRadius;
  const x2 = mouseEvent.offsetX - originX;
  const y2 = mouseEvent.offsetY - originY;
  const x1 = x2 - mouseEvent.movementX;
  const y1 = y2 - mouseEvent.movementY;
  const theta2 = Math.atan2(y2 / radius, x2 / radius);
  const theta1 = Math.atan2(y1 / radius, x1 / radius);
  let thetadiff = subtractAngle(theta2, theta1);
  moveBetLocBy(thetadiff);
}

function currentMouseTargets() {
  return [mouseTarget];
}

function mouseDidClickCanvas() {
  if (mouseTracker.isTrackingMouseMovement) {
    commitUserResponseIfReady();
  } else {
    mouseTracker.startTrackingMouseMovement();
  }
}

function handleWheel(event) {
  if (!stateAllowsPaddleControl(currentState)) {
    // Use default scrolling behavior when not currently performing a task block 
    return;
  }
  // Use wheel to change paddle size during the task
  if (event.deltaY > 0) { // wheel up
    useLargeBetWid();
    event.preventDefault();
  } else if (event.deltaY < 0) { // wheel down
    useSmallBetWid();
    event.preventDefault();
  }
}

function toggleMouseTracking() {
  if (mouseTracker.isTrackingMouseMovement) {
    mouseTracker.stopTrackingMouseMovement();
  } else {
    mouseTracker.startTrackingMouseMovement();
  }
}

function mouseDidClickWindow(event) {
  if (currentState == TaskState.BETWEEN_BLOCKS) {
    startNextBlock();
  }
}


// Keyboard

const SLIDER_MOVE_SPEED = 90 * TWOPI / 360 / 1000; // radians per millisecond
const SLIDER_MOVE_DELTA = 1000 / 60; // the slider position will be incremented every delta ms
const SLIDER_MOVE_AMOUNT = SLIDER_MOVE_SPEED * SLIDER_MOVE_DELTA;
let keyDownIntervalId;
let keyDownIntervalEventCode;

function handleKeyDown(event) {
  if (event.repeat) {
    // Do not handle the same keydown event more than once if key is being held down.
    // Events that should keep happenning while the key is being held down
    // (such as continuous slider movements) are handled via setInterval().
    return; 
  }

  switch (event.code) {
    case "KeyW":
    case "ArrowUp":
      if (stateAllowsPaddleControl(currentState)) {
        useLargeBetWid();
      }
      break;
    case "KeyS":
    case "ArrowDown":
      if (stateAllowsPaddleControl(currentState)) {
        useSmallBetWid();
      }
      break;
    case "KeyA":
    case "ArrowLeft":
      if (stateAllowsPaddleControl(currentState)) {
        clearKeyDownIntervalIfNeeded();
        keyDownIntervalEventCode = event.code
        keyDownIntervalId = setInterval(() => moveBetLocBy(-SLIDER_MOVE_AMOUNT),
          SLIDER_MOVE_DELTA);
      }
      break;
    case "KeyD":
    case "ArrowRight":
      if (stateAllowsPaddleControl(currentState)) {
        clearKeyDownIntervalIfNeeded();
        keyDownIntervalEventCode = event.code
        keyDownIntervalId = setInterval(() => moveBetLocBy(+SLIDER_MOVE_AMOUNT),
          SLIDER_MOVE_DELTA);
      }
      break;
    case "Space":
      if (currentState == TaskState.BLOCK) {
        commitUserResponseIfReady();
      } else if (currentState == TaskState.BETWEEN_BLOCKS) {
        startNextBlock();
      }
      break;
  }
}

function handleKeyUp(event) {
  if (event.code == keyDownIntervalEventCode) {
    clearKeyDownIntervalIfNeeded();
  }
}

function clearKeyDownIntervalIfNeeded() {
  if (keyDownIntervalId) {
    clearInterval(keyDownIntervalId);
  }
}

// 
// Task logic, trial phases
//

function startNextBlock() {
  if (currBlockIdx === undefined) {
    currBlockIdx = 0;
  } else {
    currBlockIdx += 1;
  }
  // initialize new block data
  let seq = seqData["seqs"][seqIndexPerBlock[currBlockIdx]];
  const nTrials = getNTrials();
  currBlockData = new BlockData(subjectId, currBlockIdx,
    seq["obs"], seq["hmean"], seq["hsd"], nTrials);
  currBlockEventData = new BlockEventData(subjectId, currBlockIdx);
  // record start time of the block
  const tStart = window.performance.now();
  currBlockEventData.recordEvent("startBlock", currBlockIdx, tStart);
  expEventData.recordEvent("startBlock", currBlockIdx, tStart);
  // start the task block
  currentState = TaskState.BLOCK;
  startInitialTrial();
}

function getNTrials() {
  const nObs = seqData["nobs"];
  return urlParams.doShortBlock ? 3 : nObs;
}

function startInitialTrial() {
  currTrialIndex = 0;
  currBlockEventData.recordEvent("startTrial", currTrialIndex);
  resetBet();
  taskDisplay.scoreDisplay.resetScore();
  taskDisplay.hideTarget();
  textbox.hidden = true;
  canvas.hidden = false;
  mouseTracker.enableTrackingUpdates();
  moveToReadyPhase();
}

function moveToNextTrial() {
  taskDisplay.hideLaser();
  taskDisplay.hideWinOutcome();
  isReady = false;
  setTimeout(moveToReadyPhase, READY_PHASE_DELAY);
  mouseTracker.enableTrackingUpdates();
  currTrialIndex += 1;
  currBlockEventData.recordEvent("startTrial", currTrialIndex);
}

async function finishBlock() {
  taskDisplay.hideLaser();
  taskDisplay.hideWinOutcome();
  mouseTracker.stopTrackingMouseMovement();
  isReady = false;
  currentState = TaskState.LOADING;
  const tEnd = window.performance.now();
  currBlockEventData.recordEvent("finishBlock", currBlockIdx, tEnd);
  expEventData.recordEvent("finishBlock", currBlockIdx, tEnd);
  const blockScore = getBlockScore();
  currBlockData.score = blockScore;
  currBlockData.pounds = pointsToPounds(blockScore);
  totalTaskScore += blockScore;
  displayEndOfBlockText();
  textbox.showSavingData();
  // Add delays between successive upload of data files to make sure they are saved
  // correctly on the Pavlovia server. For an unknown reason, without delay, the 
  // files are sometimes not recorded correctly on the server;
  // using a delay allows us to solve this issue.
  await saveDataObj(currBlockData);
  await saveDataObj(currBlockEventData, DATA_SAVE_DELAY); 
  subjectData.recordBlockDataFiles(currBlockData.jsonFilename,
    currBlockEventData.jsonFilename);
  if (currBlockIdx == nBlocks - 1) {
    subjectData.totalScore = totalTaskScore;
    subjectData.totalPounds = pointsToPounds(totalTaskScore);
    subjectData.expEventDataFile = expEventData.jsonFilename;
    await saveDataObj(subjectData, DATA_SAVE_DELAY);
    await saveDataObj(expEventData, DATA_SAVE_DELAY);

    pavloviaManager.finishServerSessionIfNeeded();

    if (urlParams.isProlificSubject) {
      textbox.showProlificCompletionLink(PROLIFIC_COMPLETION_URL);
    }

    if (urlParams.getFeedback) {
      getFeedbackForm().hidden = false;
    }
  }
  textbox.hideSavingData();
  currentState = (currBlockIdx == (nBlocks - 1) ?
    TaskState.FINISHED :
    TaskState.BETWEEN_BLOCKS);
}

function displayEndOfBlockText(){
  textbox.setBlocksCompleted(currBlockIdx+1);
  const blockScore = getBlockScore();
  textbox.setPoints(blockScore);
  textbox.setPounds(pointsToPounds(blockScore));
  if (currBlockIdx == nBlocks - 1) {
    textbox.setTotalPoints(totalTaskScore);
    textbox.setTotalPounds(Math.min(pointsToPounds(totalTaskScore), MAX_BONUS_POUNDS));
    textbox.showFinalBlockText();
  } else {
    textbox.showNextBlockText();
  }
  canvas.hidden = true;
  textbox.hidden = false;
}

function moveToReadyPhase() {
  isReady = true;
  taskDisplay.showReadyCue();
}

function commitUserResponseIfReady() {
  if (isReady) {
    commitUserResponse();
  }
}

function commitUserResponse() {
  currBlockEventData.recordEvent("commitResponse", currTrialIndex);
  mouseTracker.disableTrackingUpdates();
  isReady = false;
  const [betloc, betwid] = getBetLocWidDeg();
  currBlockData.recordResponse(currTrialIndex, betloc, betwid);
  clearKeyDownIntervalIfNeeded();
  moveToOutcomePhase();
}

const OUTCOME_PHASE_DURATION = 666; // ms
const LASER_DURATION = OUTCOME_PHASE_DURATION; //ms
const READY_PHASE_DELAY = 334; // ms

function moveToOutcomePhase() {
  // const nObs = seqData["nobs"];
  const laserLocation = currBlockData["obs"][currTrialIndex] * DEG_TO_RAD;
  const [betloc, betwid] = getBetLocWidRad();
  const absAngleLaserToSlider = Math.abs(
    subtractAngle(laserLocation, betloc));
  const didCatchLaser =  (
    Math.abs(subtractAngle(laserLocation, betloc))
    <= (betwid/2));

  taskDisplay.showLaser(laserLocation);
  taskDisplay.showWinOutcome(didCatchLaser);
  currBlockEventData.recordEvent("showWinOutcome", didCatchLaser);
  setTimeout(() => { taskDisplay.hideLaser(); }, LASER_DURATION);
  const nTrials = getNTrials();
  if (currTrialIndex == nTrials - 1) {
    setTimeout(finishBlock, OUTCOME_PHASE_DURATION);
  } else {
    setTimeout(moveToNextTrial, OUTCOME_PHASE_DURATION);
  }
}

function resetBet() {
  currBetWidIdx = INITIAL_VALUES.betWidIdx;
  taskDisplay.slider.location = INITIAL_VALUES.betLoc;
  taskDisplay.slider.widthIdx = currBetWidIdx;
}

function getBetLocRad() {
  return taskDisplay.slider.location;
}

function getBetLocDeg() {
  return getBetLocRad() * RAD_TO_DEG;
}

function getBetWidRad() {
  return BET_WIDTHS_RAD[currBetWidIdx];
}

function getBetWidDeg() {
  return getBetWidRad() * RAD_TO_DEG;
}

function getBetLocWidRad() {
  return [getBetLocRad(), getBetWidRad()];
}

function getBetLocWidDeg() {
  return [getBetLocDeg(), getBetWidDeg()];
}

function useSmallBetWid() {
  // currBlockData.recordEvent("useSmallBetWid", null);
  useBetWidIdx(0);
}

function useLargeBetWid() {
  // currBlockData.recordEvent("useLargeBetWid", null);
  useBetWidIdx(1);
}

function useBetWidIdx(idx) {
  const safeIdx = idx % BET_WIDTHS_RAD.length;
  if (currBetWidIdx != safeIdx) {
    currBetWidIdx = safeIdx;
    taskDisplay.slider.widthIdx = currBetWidIdx;
    currBlockEventData?.recordEvent("changeBetWid", getBetWidDeg());
  }

  // For the practice phase: check if the correct width was used and complete
  // the practice if so
  if ((currentState == TaskState.PRACTICE_SWITCH_LARGE_WID)
    && (currBetWidIdx == 1)) {
    // Practice switch large width completed
    setTimeout(() => {
      //  move to practice switch small width
      currentState = TaskState.PRACTICE_SWITCH_SMALL_WID;
      taskDisplay.practiceText = PRACTICE_TEXT.SWITCH_SMALL_WID;
    }, PRACTICE_DELAY);
  } else if ((currentState == TaskState.PRACTICE_SWITCH_SMALL_WID)
    && (currBetWidIdx == 0)) {
    // Practice switch small width completed"
    setTimeout(() => {
      // move to next instruction
      currentState = TaskState.INSTRUCTIONS;
      goToInstructionIdx(PRACTICE_SWITCH_WID_PRECEDING_INSTRUCTION_IDX + 1);
      instructionContainer.hidden = false;
      consentContainer.hidden = true;
      canvas.hidden = true;
      mouseTracker.stopTrackingMouseMovement();
      practiceHelpPanel.hidden = true;
      taskDisplay.practiceText = null;
    }, PRACTICE_DELAY);
  }
}

function moveBetLocBy(radians) {
  taskDisplay.slider.location = ((taskDisplay.slider.location + radians)
        % TWOPI);
  currBlockEventData?.recordEvent("moveBetLoc", getBetLocDeg());

  // For the practice phase: Check if the target was reached and complete the
  // practice if so
  if ((currentState == TaskState.PRACTICE_MOVE_LOCATION_1)
    && betLocIsCloseTo(getBetLocRad(), PRACTICE_TARGET_LOCATION_1)) {
    setTimeout(() => {
      // move to practice location #2
      currentState = TaskState.PRACTICE_MOVE_LOCATION_2;
      // TBD: update designated location
      taskDisplay.practiceText = PRACTICE_TEXT.TARGET_LOCATION;
      taskDisplay.showTarget(PRACTICE_TARGET_LOCATION_2);
    }, PRACTICE_DELAY);
    
  } else if ((currentState == TaskState.PRACTICE_MOVE_LOCATION_2)
    && betLocIsCloseTo(getBetLocRad(), PRACTICE_TARGET_LOCATION_2)) {
    setTimeout(() => {
      // move to next instruction
      currentState = TaskState.INSTRUCTIONS;
      goToInstructionIdx(PRACTICE_MOVE_LOCATION_PRECEDING_INSTRUCTION_IDX + 1);
      instructionContainer.hidden = false;
      consentContainer.hidden = true;
      canvas.hidden = true;
      mouseTracker.stopTrackingMouseMovement();
      practiceHelpPanel.hidden = true;
      taskDisplay.practiceText = null;
      taskDisplay.hideTarget();
    }, PRACTICE_DELAY);
  }
}

function betLocIsCloseTo(loc, target) {
  return (Math.abs(subtractAngle(loc, target))
    <= (SLIDER_MOVE_AMOUNT * 2));
}

function getBlockScore() {
  return taskDisplay.scoreDisplay.score;
}

function pointsToPounds(pts) {
  const nPointsPerPound = urlParams.nPointsPerPound ?? N_POINTS_PER_POUND;
  return pts / nPointsPerPound;
}

//
// Task instructions
//

function loadAndStartInstructions() {
  // Fetch instructions and start displaying the instructions
  loadInstructions(startInstructions, startOrResumeTaskFromInstructions,
    currInstructionWillChange, currInstructionDidChange, quizDidSubmitAnwers,
    nBlocks, urlParams.isProlificSubject);
}

function startInstructions() {
  currentState = TaskState.INSTRUCTIONS;
  instructionContainer.hidden = false;
  consentContainer.hidden = true;
  canvas.hidden = true;
  textbox.hidden = true;
  helpPanel.hidden = true;
  expEventData.recordEvent("startInstructions", null);
  // Skip to last instruction if the 'skipInstruction' URL parameter is set
  if (!urlParams.doInstructions) {
    goToInstructionIdx(n_instructions - 1);
  }
}

function startOrResumeTaskFromInstructions() {
  currentState = TaskState.BLOCK;
  instructionContainer.hidden = true;
  consentContainer.hidden = true;
  taskDisplay.hideTarget();
  canvas.hidden = false;
  helpPanel.hidden = false;
  startNextBlock();
}

function currInstructionWillChange(prevIdx, nextIdx) {
  if ((prevIdx == PRACTICE_MOVE_LOCATION_PRECEDING_INSTRUCTION_IDX)
    && (nextIdx == (PRACTICE_MOVE_LOCATION_PRECEDING_INSTRUCTION_IDX + 1))) {
    // Move to practice location phase
    currentState = TaskState.PRACTICE_MOVE_LOCATION_1;
    instructionContainer.hidden = true;
    consentContainer.hidden = true;
    taskDisplay.showTarget(PRACTICE_TARGET_LOCATION_1);
    canvas.hidden = false;
    practiceHelpPanel.hidden = false;
    taskDisplay.practiceText = PRACTICE_TEXT.TARGET_LOCATION;
  } else if ((prevIdx == PRACTICE_SWITCH_WID_PRECEDING_INSTRUCTION_IDX)
    && (nextIdx == (PRACTICE_SWITCH_WID_PRECEDING_INSTRUCTION_IDX + 1))) {
    // Move to practice switch width phase
    currentState = TaskState.PRACTICE_SWITCH_LARGE_WID;
    instructionContainer.hidden = true;
    consentContainer.hidden = true;
    taskDisplay.hideTarget();
    canvas.hidden = false;
    practiceHelpPanel.hidden = false;
    taskDisplay.practiceText = PRACTICE_TEXT.SWITCH_LARGE_WID;
  }
}

function currInstructionDidChange() {
  expEventData.recordEvent("viewInstruction", currInstructionIdx);
}

function quizDidSubmitAnwers(answers) {
  expEventData.recordEvent("submitQuizAnswers", answers)
}

//
// Informed consent
//

function loadAndStartConsent() {
  loadConsent(startConsent, didAcceptConsent);
}

function startConsent() {
  currentState = TaskState.CONSENT;
  consentContainer.hidden = false;
  instructionContainer.hidden = true;
  canvas.hidden = true;
  textbox.hidden = true;
  helpPanel.hidden = true;
  expEventData.recordEvent("startConsent", null);
}

function didAcceptConsent() {
  expEventData.recordEvent("didAcceptConsent", null);
  loadAndStartInstructions();
}

//
// Pavlovia server:
// Code to upload the subject's data to the Pavlovia server.
//

let pavloviaManager = new PavloviaManager(saveIncompleteData);

// Note: saveDataObj will save to the server if the host is Pavlovia, and
// otherwise will save to the local file system (the browser will either prompt
// the user to download the file, or the download will be trigerred automatically).
async function saveDataObj(dataObj, delay = null) {
  let didSaveSucceed;
  try {
    await pavloviaManager.saveData(
      dataObj.asJSONString, dataObj.jsonFilename,
      delay);
    didSaveSucceed = true;
  } catch (error) {
    didSaveSucceed = false;
  }

  return didSaveSucceed;
}

// Note: This function is called when the user closes the window before the
// experiment is complete. It is still to be tested: I am not sure the
// data is successfully saved to the server when that happens.
function saveIncompleteData() {
  console.log("saving incomplete data");
  expEventData.recordEvent("saveIncompleteData", currBlockIdx);
  subjectData.expEventDataFile = expEventData.jsonFilename;
  if (currBlockData && currBlockEventData) {
    currBlockData.nTrials = (currTrialIndex + 1);
    subjectData.recordBlockDataFiles(currBlockData.jsonFilename,
      currBlockEventData.jsonFilename);
  }
  for (let dataObj of [expEventData, subjectData,
    currBlockData, currBlockEventData]) {
    if (dataObj) {
      saveDataObj(dataObj);
    }
  }
}

// 
// Initialize global variables and perform the initial setup
// 
let canvas = getCanvas();
canvas.adaptToScreenDPI();
canvas.hidden = (currentState != TaskState.BLOCK
  && !stateIsPractice(currentState));

let centerX = canvas.widthCSS / 2;
let centerY = canvas.heightCSS / 2;
let taskDisplay = new TaskDisplay(centerX, centerY, canvas.widthCSS);

let currBetWidIdx;

let mouseTracker = new MouseTracker(
    updateForMouseMovement,
    currentMouseTargets);
mouseTracker.domElement = canvas;

let mouseTarget = new MouseTarget(mouseDidClickCanvas);

let helpPanel = getHelpPanel();
helpPanel.hidden = (currentState != TaskState.BLOCK
  && (currentState != TaskState.BETWEEN_BLOCKS));

let practiceHelpPanel = getPracticeHelpPanel();
practiceHelpPanel.hidden = (!stateIsPractice(currentState))

let textbox = getTextBox();
textbox.hidden = ((currentState != TaskState.BETWEEN_BLOCKS)
  && (currentState != TaskState.FINISHED));
textbox.setTotalBlocks(nBlocks);

let instructionContainer = getInstructionContainer();
instructionContainer.hidden = currentState != TaskState.INSTRUCTIONS;

let consentContainer = getConsentContainer();
instructionContainer.hidden = currentState != TaskState.CONSENT;

window.addEventListener("click", mouseDidClickWindow);
window.addEventListener("wheel", handleWheel, { passive: false }); 
window.addEventListener("keydown", handleKeyDown, true);
window.addEventListener("keyup", handleKeyUp, true);

setNeedsRedraw();

// Fetch the stimulus sequences data and initialize the subject data
async function fetchSeqData() {
  // fetch sequence data
  let response = await fetch(seq_data_path);
  seqData = await response.json();
  // Assign a stimulus sequence to each block of the experiment
  // by randomly sampling nBlocks sequences (without replacement)
  // from the sequence dataset.
  seqIndexPerBlock = randomSample(range(seqData["nseq"]), nBlocks);
}
fetchSeqData();

// Start session with the Pavlovia server
pavloviaManager.initServerSessionIfNeeded();


// Start at the consent phase or the instruction phase
if (urlParams.doConsent) {
  loadAndStartConsent();
} else {
  loadAndStartInstructions();
}