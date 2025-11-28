import {range} from './utils/num.js';
import {version} from './version.js';

// 
// Class definitions
// 

export class JSONSavableData {
  get asJSONString() {
    return objToJSONString(this);
  }
}

export class BlockData extends JSONSavableData {
  constructor(
    subjectId,
    blockIdx,
    obs,
    hmean,
    hsd,
    nTrials
    ) {
    super();
    this.subjectId = subjectId;
    this.version = version;
    this.blockIdx = blockIdx;
    this.obs = obs;
    this.hmean = hmean;
    this.hsd = hsd;
    this.nTrials = nTrials;
    this.trialIdx = range(nTrials);
    this.betloc = new Array(nTrials);
    this.betwid = new Array(nTrials);
  }

  recordResponse(trialIdx, betloc, betwid) {
    this.betloc[trialIdx] = betloc;
    this.betwid[trialIdx] = betwid;
  }

  get jsonFilename() {
    const fbasenameParts = [
      'infer-bet', // old name for 'infer-pred', kept for compatibility
      `version-${this.version}`,
      `sub-${this.subjectId}`,
      `block-${this.blockIdx+1}`];
    const fbasename = fbasenameParts.join("_");
    return `${fbasename}.json`;
  }
}

export class EventData extends JSONSavableData {
  constructor() {
    super();
    this.events = {};
  }

  recordEvent(type, value, time = null) {
    if (!time) {
      time = window.performance.now();
    }
    if (!(type in this.events)) {
      // initialize empty array for this new event type
      this.events[type] = new Array();
    }
    this.events[type].push([time, value]);
  }
}

// Events occurring within a task block
export class BlockEventData extends EventData {
  constructor(subjectId, blockIdx) {
    super();
    this.subjectId = subjectId;
    this.version = version;
    this.blockIdx = blockIdx;
  }

  get jsonFilename() {
    const fbasenameParts = [
      'infer-bet', // old name for 'infer-pred', kept for compatibility
      `version-${this.version}`,
      'block-events',
      `sub-${this.subjectId}`,
      `block-${this.blockIdx+1}`];
    const fbasename = fbasenameParts.join("_");
    return `${fbasename}.json`;
  }
}

// Events occurring outside or between blocks,
// such as consent acceptance or instruction viewing
export class ExperimentEventData extends EventData {
  constructor(subjectId) {
    super();
    this.subjectId = subjectId;
    this.version = version;
  }

  get jsonFilename() {
    const fbasenameParts = [
      'infer-bet', // old name for 'infer-pred', kept for compatibility
      `version-${this.version}`,
      'exp-events',
      `sub-${this.subjectId}`];
    const fbasename = fbasenameParts.join("_");
    return `${fbasename}.json`;
  }
}

export class SubjectData extends JSONSavableData {
  constructor(subjectId) {
    super();
    this.subjectId = subjectId;
    this.version = version;
    this.expEventDataFile = null;
    this.blockDataFiles = [];
    this.blockEventDataFiles = [];
  }

  recordBlockDataFiles(blockDataFile, blockEventDataFile) {
    this.blockDataFiles.push(blockDataFile);
    this.blockEventDataFiles.push(blockEventDataFile);
  }

  get jsonFilename() {
    const fbasenameParts = [
      'infer-bet', // old name for 'infer-pred', kept for compatibility
      `version-${this.version}`,
      `sub-data`,
      `sub-${this.subjectId}`];
    const fbasename = fbasenameParts.join("_");
    return `${fbasename}.json`;
  }
}

//
// Function definitions
//

function promptToSaveObjectAsJSONFile(obj, fname, space=null) {
  const jsonString = objToJSONString(obj, space);
  promptToSaveStrToFile(jsonString, fname, 'application/json');
}

export function objToJSONString(obj, space=null) {
  return JSON.stringify(obj, null, space);
}

export function promptToSaveStrToFile(str, fname, type = 'text') {
  let blob = new Blob( [ str ], { type });
  let url = URL.createObjectURL( blob );
  let link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', fname);
  let event = document.createEvent( 'MouseEvents' );
  event.initMouseEvent('click', true, true, window, 1, 0, 0, 0, 0, false, false, false, false, 0, null);
  link.dispatchEvent(event);
}
