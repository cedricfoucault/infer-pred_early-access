const CONSENT_HTML_PATH = "consent.html";

export async function loadConsent(onloadCallback, acceptHandler,
  containerId = "consent-container") {
  // Fetch HTML content to embed in the container
  let response = await fetch(CONSENT_HTML_PATH);
  let htmlContent = await response.text();
  // Embed the HTML content within the container in the document tree 
  const container = document.getElementById(containerId);
  container.innerHTML = htmlContent;
  // Perform the initial setup to be performed at runtime;
  setupConsent(acceptHandler);
  // Run the callback once the html has been loaded and set up on the document tree
  if (onloadCallback) {
    onloadCallback();
  }
}

function setupConsent(acceptHandler) {
  if (acceptHandler) {
    document.getElementById("accept-consent-button").onclick = acceptHandler;
  }
}
