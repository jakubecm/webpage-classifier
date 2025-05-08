chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "classify") {
    fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ html: message.html })
    })
    .then(res => res.json())
    .then(data => {
      chrome.runtime.sendMessage({
        action: "showResult",
        prediction: data.prediction
      });
    })
    .catch(err => console.error("Error:", err));
  }
});
