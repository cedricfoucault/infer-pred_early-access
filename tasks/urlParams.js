export function getURLParams() {
  return new URLSearchParams(window.location.search);
}

export function parseURLParams() {
  const urlParams = getURLParams();

  urlParams.subjectId = urlParams.get('subjectId');
  // - Whether to get subject's informed consent for participating in the experiment
  urlParams.doConsent = (urlParams.has('skipConsent') ? false : true);
  // - Whether to show the task instructions
  urlParams.doInstructions = (urlParams.has('skipInstructions') ? false : true);
  // - Whether to show the form to get participants' feedback
  urlParams.getFeedback = (urlParams.has('getFeedback') ? true : false);
  // - Whether to shorten the blocks to just a few trials (useful for debugging the task)
  urlParams.doShortBlock = (urlParams.has('shortBlock') ? true : false);
  // - Number of sessions to perform for each task
  urlParams.nBlocks = urlParams.get('nBlocks');
  // - Money bonus : number of points to earn £1
  urlParams.nPointsPerPound = urlParams.get('nPointsPerPound');

  // Prolific information
  urlParams.isProlificSubject = urlParams.has('PROLIFIC_PID');
  urlParams.prolificPid = urlParams.get('PROLIFIC_PID');
  urlParams.studyId = urlParams.get('STUDY_ID');
  urlParams.sessionId = urlParams.get('SESSION_ID');

  // If the subejctID has not been set, set it to th prolificID if it exists
  if (urlParams.isProlificSubject && (!urlParams.has('subjectId'))) {
    urlParams.subjectId = urlParams.prolificPid;
  }

  // - Version of the task ('rw' or 'cp')
  urlParams.version = urlParams.get('version');

  return urlParams;
}
