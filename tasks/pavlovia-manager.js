import {pavloviaConnector} from './utils/pavlovia-connector.js';
import {promptToSaveStrToFile} from './subjectData.js';

export function isHostPavlovia() {
  const hostname = window.location.hostname;
  return hostname.includes("pavlovia");
}

export class PavloviaManager {
  constructor(saveIncompleteResultsCallback) {
    this.shouldConnectToPavloviaServer = isHostPavlovia();
    this.pavloviaConnector = pavloviaConnector;
    this.saveIncompleteResultsCallback = saveIncompleteResultsCallback;
    this.hasServerSession = false;
  }

  async initServerSessionIfNeeded() {
    if (this.shouldConnectToPavloviaServer
      && !this.hasServerSession) {
      return await this.initServerSession();
    }
  }

  async finishServerSessionIfNeeded() {
    if (this.shouldConnectToPavloviaServer
      && this.hasServerSession) {
      return await this.finishServerSession();
    }
  }

  async initServerSession() {
    console.log("PavloviaManager.initServerSession called");
    let response = this.pavloviaConnector._init('config.json',
      this.saveIncompleteResultsCallback);
    this.hasServerSession = true;
    return await response;
  }

  async finishServerSession() {
    console.log("PavloviaManager.finishServerSession called");
    let response = this.pavloviaConnector._finish();
    this.hasServerSession = false;
    return await response;
  }

  async saveData(data, filename, delay = null, sync = false, type = 'application/json') {
    // save the data to the server or locally depending on the current host
    if (delay) {
      await sleep(delay);
    }
    if (this.shouldConnectToPavloviaServer) {
      try {
        await this.saveDataToServer(data, filename, sync);
      }
      catch (error) {
        console.error('PavloviaManager.saveDataToServer failed with error',
          error);
        throw error
      }
    } else {
      promptToSaveStrToFile(data, filename, type);
    }
  }

  async saveDataToServer(data, filename,
    sync = false, type = 'application/json') {
    console.log("PavloviaManager.saveDataToServer called");
    if (!this.hasServerSession) {
      let errorMsg = "A server session must be initiated \
        before any data can be saved to the server";
      console.error(errorMsg);
      throw errorMsg;
    }
    return await this.pavloviaConnector._save(data, filename, sync, type);
  }
}

// ref https://stackoverflow.com/questions/951021/what-is-the-javascript-version-of-sleep
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

