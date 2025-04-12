document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("csv-file");
  if (!fileInput.files.length) return alert("Select a CSV file.");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  try {
    const res = await fetch("/upload_csv", { method: "POST", body: formData });
    const data = await res.json();
    document.getElementById("upload-result").innerText = `${data.detail} (${data.num_rows} rows)`;
    loadCategories();
    loadSentimentStats();
    loadCategoryDistribution();
  } catch (err) {
    console.error("Upload error:", err);
    document.getElementById("upload-result").innerText = "⚠️ CSV upload failed.";
  }
});

async function loadCategories() {
  try {
    const res = await fetch("/categories");
    if (!res.ok) throw new Error("Failed to fetch cluster categories");
    const data = await res.json();
    const select = document.getElementById("category-select");
    select.innerHTML = "";
    if (!data.categories || !data.clusters) {
      console.error("Invalid /categories response:", data);
      return;
    }
    data.categories.forEach((label, i) => {
      const option = document.createElement("option");
      option.value = data.clusters[i];       // Cluster number as value
      option.textContent = label;            // GPT-generated label
      select.appendChild(option);
    });
    console.log("✅ Categories loaded:", data.categories);
  } catch (err) {
    console.error("❌ Error loading categories:", err);
    document.getElementById("upload-result").innerText =
      "⚠️ Failed to load categories. Please re-upload the CSV.";
  }
}

async function loadSentimentStats() {
  try {
    const res = await fetch("/sentiment_stats");
    const data = await res.json();
    renderDonutChart(data.labels, data.data, "sentiment-chart", "Review Sentiment");
    const box = document.getElementById("sample-comments-container");
    box.innerHTML = "";
    for (const [label, samples] of Object.entries(data.samples)) {
      const div = document.createElement("div");
      div.innerHTML = `<strong>${label}</strong><ul>` + samples.map(c => "<li>" + c + "</li>").join("") + "</ul>";
      box.appendChild(div);
    }
  } catch (err) {
    console.error("Sentiment stats failed:", err);
  }
}

async function loadCategoryDistribution() {
  try {
    const res = await fetch("/category_distribution");
    const data = await res.json();
    renderBarChart(data.labels, data.data, "% of Reviews per Category");
  } catch (err) {
    console.error("Category distribution error:", err);
  }
}

document.getElementById("classification-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = document.getElementById("review-input").value;
  try {
    const res = await fetch("/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review: text }),
    });
    const data = await res.json();
    document.getElementById("classification-result").innerText = "Classification: " + data.result;
  } catch (err) {
    console.error("Classification error:", err);
    document.getElementById("classification-result").innerText = "⚠️ Error classifying review.";
  }
});

document.getElementById("summarization-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const select = document.getElementById("category-select");
  const clusterId = select.value;

  if (!clusterId || clusterId === "undefined") {
    alert("Please select a cluster category.");
    return;
  }

  const resultBox = document.getElementById("summarization-result");
  // Check if a rating chart element exists before using it
  const chartCanvas = document.getElementById("rating-chart");
  
  try {
    console.log("📤 Sending cluster ID:", clusterId);
    const res = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category: clusterId })
    });
    if (!res.ok) {
      const err = await res.json();
      console.error("❌ /summarize error response:", err);
      resultBox.innerText = "⚠️ " + (err.detail || "Summary request failed.");
      if (chartCanvas) chartCanvas.style.display = "none";
      return;
    }
    const data = await res.json();
    console.log("✅ /summarize returned:", data);

    if (!data.summary || !data.category) {
      console.warn("⚠️ Summary data missing in response:", data);
      resultBox.innerText = "⚠️ Summary data missing. Please check the server logs or try another cluster.";
      if (chartCanvas) chartCanvas.style.display = "none";
      return;
    }

    // Display the summary on the page.
    resultBox.innerText = `🧠 Summary for ${data.category}:\n\n${data.summary}`;

    if (data.top3_products && data.ratings && data.ratings.length > 0) {
      if (chartCanvas) {
        chartCanvas.style.display = "block";
        renderDonutChart(data.top3_products, data.ratings, "rating-chart", "Top Product Ratings");
      }
    } else {
      console.warn("⚠️ No chart data.");
      if (chartCanvas) chartCanvas.style.display = "none";
    }

  } catch (err) {
    console.error("❌ Fetch error:", err);
    resultBox.innerText = "Error generating summary. Please try again later.";
    if (chartCanvas) chartCanvas.style.display = "none";
  }
});

let chartMap = {};

function renderBarChart(labels, data, canvasId, labelText) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  if (chartMap[canvasId]) chartMap[canvasId].destroy();
  chartMap[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: labelText,
        data: data,
        backgroundColor: "#4caf50"
      }]
    },
    options: {
      responsive: true,
      indexAxis: 'y',
      scales: {
        x: {
          beginAtZero: true,
          ticks: { stepSize: 10 }
        }
      }
    }
  });
}

function renderDonutChart(labels, data, canvasId, labelText) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  if (chartMap[canvasId]) chartMap[canvasId].destroy();
  chartMap[canvasId] = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: labels,
      datasets: [{
        label: labelText,
        data: data,
        backgroundColor: ["#4caf50", "#2196f3", "#ff9800"],
        borderColor: "#fff",
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: function(context) {
              return context.label + ": " + context.formattedValue;
            }
          }
        }
      }
    }
  });
}
