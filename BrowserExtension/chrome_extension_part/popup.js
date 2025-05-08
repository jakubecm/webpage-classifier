document.getElementById("classifyBtn").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      function: () => {
        chrome.runtime.sendMessage({
          action: "classify",
          html: document.documentElement.outerHTML
        });
      }
    });
  });

  document.getElementById("result").textContent = "Classifying...";
});

chrome.runtime.onMessage.addListener((message) => {
  if (message.action === "showResult") {
    document.getElementById("result").textContent = message.prediction;
  }
});
