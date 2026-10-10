const API = "";

// ================================
// PAGE ELEMENTS
// ================================

const loginPage = document.getElementById("loginPage");
const registerPage = document.getElementById("registerPage");
const dashboardPage = document.getElementById("dashboardPage");


// ================================
// PAGE SWITCH
// ================================

function showLogin() {
    loginPage.classList.remove("hidden");
    registerPage.classList.add("hidden");
    dashboardPage.classList.add("hidden");
}

function showRegister() {
    loginPage.classList.add("hidden");
    registerPage.classList.remove("hidden");
    dashboardPage.classList.add("hidden");
}

function showDashboard() {
    loginPage.classList.add("hidden");
    registerPage.classList.add("hidden");
    dashboardPage.classList.remove("hidden");
}


// ================================
// LOGIN
// ================================

document.getElementById("loginForm").addEventListener("submit", async function(e) {

    e.preventDefault();

    const username = document.getElementById("loginUsername").value;
    const password = document.getElementById("loginPassword").value;

    const message = document.getElementById("loginMessage");

    message.textContent = "Logging in...";

    try {

        const formData = new URLSearchParams();

        formData.append("username", username);
        formData.append("password", password);

        const response = await fetch(`${API}/auth/login`, {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded"
            },
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Login failed");
        }

        localStorage.setItem("access_token", data.access_token);

        localStorage.setItem(
            "username",
            username
        );

        document.getElementById("currentUser").textContent = username;

        message.textContent = "";

        showDashboard();

        showSection("askSection");

    } catch (error) {

        message.textContent = error.message;
        message.style.color = "red";

    }

});


// ================================
// REGISTER
// ================================

document.getElementById("registerForm").addEventListener("submit", async function(e) {

    e.preventDefault();

    const username =
        document.getElementById("registerUsername").value;

    const email =
        document.getElementById("registerEmail").value;

    const password =
        document.getElementById("registerPassword").value;

    const message =
        document.getElementById("registerMessage");

    message.textContent = "Creating account...";

    try {

        const response = await fetch(`${API}/auth/register`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                name: username,
                email: email,
                password: password
            })

        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Registration failed"
            );
        }

        message.style.color = "green";

        message.textContent =
            "Registration successful. Please login.";

        document.getElementById("registerForm").reset();

        setTimeout(() => {
            showLogin();
        }, 1200);

    } catch (error) {

        message.style.color = "red";
        message.textContent = error.message;

    }

});


// ================================
// AUTH HEADER
// ================================

function authHeaders() {

    const token =
        localStorage.getItem("access_token");

    return {
        "Authorization": `Bearer ${token}`
    };

}


// ================================
// LOGOUT
// ================================

function logout() {

    localStorage.removeItem("access_token");
    localStorage.removeItem("username");

    showLogin();

}


// ================================
// SECTION SWITCH
// ================================

function showSection(sectionId) {

    document
        .querySelectorAll(".section")
        .forEach(section => {
            section.classList.add("hidden");
        });

    document
        .getElementById(sectionId)
        .classList.remove("hidden");

    if (sectionId === "historySection") {
        loadHistory();
    }

}


// ================================
// ASK AI
// ================================

async function askAI() {

    const question =
        document.getElementById("questionInput").value.trim();

    const answerBox =
        document.getElementById("answerBox");

    const answerText =
        document.getElementById("answerText");

    if (!question) {
        alert("Please enter a question.");
        return;
    }

    answerBox.classList.remove("hidden");

    answerText.textContent =
        "🤖 AI is thinking...";

    try {

        const response = await fetch(
            `${API}/rag/ask?query=${encodeURIComponent(question)}`,
            {
                method: "GET",
                headers: authHeaders()
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "AI request failed"
            );
        }

        answerText.textContent =
            data.answer || "No answer received.";

    } catch (error) {

        answerText.textContent =
            "❌ " + error.message;

    }

}


// ================================
// PDF UPLOAD
// ================================

async function uploadPDF() {

    const fileInput =
        document.getElementById("pdfFile");

    const message =
        document.getElementById("uploadMessage");

    if (!fileInput.files.length) {

        message.style.color = "red";
        message.textContent =
            "Please select a PDF file.";

        return;
    }

    const file = fileInput.files[0];

const allowedExtensions = [
    ".pdf", ".docx", ".txt", ".csv",
    ".xls", ".xlsx", ".jpg", ".jpeg", ".png"
];

const extension = file.name
    .toLowerCase()
    .slice(file.name.lastIndexOf("."));

if (!allowedExtensions.includes(extension)) {
    message.style.color = "red";
    message.textContent =
        "Supported files: PDF, DOCX, TXT, CSV, XLS, XLSX, JPG, JPEG, PNG.";
    return;
}
    const formData = new FormData();

    formData.append("file", file);

    message.style.color = "#2563eb";
    message.textContent =
        "Uploading and processing PDF...";

    try {

        const response = await fetch(
            `${API}/rag/upload`,
            {
                method: "POST",
                headers: authHeaders(),
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "PDF upload failed"
            );
        }

        message.style.color = "green";

        message.textContent =
            `✅ ${data.message} (${data.characters} characters)`;

        await loadDocuments();
        fileInput.value = "";

    } catch (error) {

        message.style.color = "red";
        message.textContent =
            "❌ " + error.message;

    }

}


// ================================
// CHAT HISTORY
// ================================

async function loadHistory() {

    const historyList =
        document.getElementById("historyList");

    historyList.innerHTML =
        "Loading history...";

    try {

        const response = await fetch(
            `${API}/history/`,
            {
                method: "GET",
                headers: authHeaders()
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Failed to load history"
            );
        }

        if (!data.history || data.history.length === 0) {

            historyList.innerHTML =
                "<p>No chat history found.</p>";

            return;
        }

        historyList.innerHTML = "";

        data.history.forEach(item => {

            const div =
                document.createElement("div");

            div.className =
                "history-item";

            div.innerHTML = `

                <div class="question">
                    ❓ ${escapeHTML(item.question)}
                </div>

                <div class="answer">
                    🤖 ${escapeHTML(item.answer)}
                </div>

                <button
                    class="delete-btn"
                    onclick="deleteHistory(${item.id})"
                >
                    🗑 Delete
                </button>

            `;

            historyList.appendChild(div);

        });

    } catch (error) {

        historyList.innerHTML =
            `<p style="color:red">❌ ${error.message}</p>`;

    }

}


// ================================
// DELETE HISTORY
// ================================

async function deleteHistory(id) {

    if (!confirm("Delete this chat history?")) {
        return;
    }

    try {

        const response = await fetch(
            `${API}/history/${id}`,
            {
                method: "DELETE",
                headers: authHeaders()
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Delete failed"
            );
        }

        loadHistory();

    } catch (error) {

        alert("❌ " + error.message);

    }

}


// ================================
// HTML SECURITY
// ================================

function escapeHTML(value) {

    const div =
        document.createElement("div");

    div.textContent =
        value ?? "";

    return div.innerHTML;

}


// ================================
// ENTER KEY → ASK AI
// ================================

document
    .getElementById("questionInput")
    .addEventListener("keydown", function(e) {

        if (e.key === "Enter") {
            askAI();
        }

    });


// ================================
// AUTO LOGIN CHECK
// ================================

window.addEventListener("load", function() {

    const token =
        localStorage.getItem("access_token");

    const username =
        localStorage.getItem("username");

    if (token) {

        document.getElementById(
            "currentUser"
        ).textContent = username || "User";

        showDashboard();

        showSection("askSection");

    } else {

        showLogin();

    }

});

// ================================
// UPLOADED DOCUMENTS
// ================================

async function loadDocuments() {
    const list = document.getElementById("documentsList");
    if (!list) return;

    list.textContent = "Loading documents...";

    try {
        const response = await fetch("/rag/documents", {
            headers: authHeaders()
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Could not load documents");
        }

        const documents = Array.isArray(data) ? data : (data.documents || []);
        list.replaceChildren();

        if (documents.length === 0) {
            list.textContent = "No documents uploaded yet.";
            return;
        }

        documents.forEach((doc) => {
            const item = document.createElement("div");
            item.className = "document-item";

            const name = document.createElement("span");
            name.textContent = `${doc.filename} (${doc.file_type || "file"})`;

            const remove = document.createElement("button");
            remove.textContent = "Delete";
            remove.type = "button";
            remove.onclick = () => deleteDocument(doc.id);

            item.append(name, remove);
            list.appendChild(item);
        });
    } catch (error) {
        list.textContent = "Error: " + error.message;
    }
}

async function deleteDocument(id) {
    if (!confirm("Are you sure you want to delete this document?")) {
        return;
    }

    try {
        const response = await fetch(`/rag/documents/${id}`, {
            method: "DELETE",
            headers: authHeaders()
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Document deletion failed");
        }

        alert(data.message || "Document deleted successfully");
        await loadDocuments();
    } catch (error) {
        alert("Error: " + error.message);
    }
}
