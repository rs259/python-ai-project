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
    sessionStorage.clear();

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
// MULTI-FORMAT DOCUMENT UPLOAD
// ================================

async function uploadPDF() {
    const fileInput = document.getElementById("pdfFile");
    const message = document.getElementById("uploadMessage");

    if (!fileInput.files.length) {
        message.style.color = "red";
        message.textContent = "Please select at least one document file.";
        return;
    }

    const allowedExtensions = [
        ".pdf", ".docx", ".txt", ".csv",
        ".xls", ".xlsx", ".jpg", ".jpeg", ".png"
    ];

    const files = Array.from(fileInput.files);

    const invalidFiles = files.filter(file => {
        const dot = file.name.lastIndexOf(".");
        const extension = dot >= 0
            ? file.name.slice(dot).toLowerCase()
            : "";
        return !allowedExtensions.includes(extension);
    });

    if (invalidFiles.length) {
        message.style.color = "red";
        message.textContent =
            "Unsupported file(s): " +
            invalidFiles.map(file => file.name).join(", ");
        return;
    }

    let succeeded = 0;
    const failed = [];

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append("file", file);

        message.style.color = "#2563eb";
        message.textContent =
            `Uploading ${i + 1}/${files.length}: ${file.name}`;

        try {
            const response = await fetch(`${API}/rag/upload`, {
                method: "POST",
                headers: authHeaders(),
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Upload failed");
            }

            succeeded++;
        } catch (error) {
            failed.push(`${file.name}: ${error.message}`);
        }
    }

    await loadDocuments();
    fileInput.value = "";

    if (failed.length === 0) {
        message.style.color = "green";
        message.textContent =
            `✅ Successfully uploaded ${succeeded} document(s).`;
    } else {
        message.style.color = "red";
        message.textContent =
            `Uploaded ${succeeded}/${files.length}. Failed: ` +
            failed.join(" | ");
    }
}

async function loadDocuments() {
    const list = document.getElementById("documentsList");

    if (!list) {
        console.error("documentsList element not found");
        return;
    }

    list.textContent = "Loading documents...";

    try {
        const response = await fetch(`${API}/rag/documents`, {
            method: "GET",
            headers: authHeaders()
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Could not load documents");
        }

        const documents = Array.isArray(data)
            ? data
            : (data.documents || []);

        list.replaceChildren();

        if (documents.length === 0) {
            list.textContent = "No documents uploaded yet.";
            return;
        }

        documents.forEach(doc => {
            const item = document.createElement("div");
            item.className = "document-item";

            const name = document.createElement("span");
            name.textContent = `${doc.filename} (${doc.file_type || "file"})`;

            const deleteButton = document.createElement("button");
            deleteButton.type = "button";
            deleteButton.textContent = "Delete";
            deleteButton.style.marginLeft = "12px";
            deleteButton.style.backgroundColor = "#dc2626";
            deleteButton.style.color = "#ffffff";

            deleteButton.addEventListener("click", async () => {
                if (!confirm(`Delete "${doc.filename}"?`)) {
                    return;
                }

                deleteButton.disabled = true;
                deleteButton.textContent = "Deleting...";

                try {
                    const deleteResponse = await fetch(
                        `${API}/rag/documents/${doc.id}`,
                        {
                            method: "DELETE",
                            headers: authHeaders()
                        }
                    );

                    const deleteData = await deleteResponse.json();

                    if (!deleteResponse.ok) {
                        throw new Error(
                            deleteData.detail || "Could not delete document"
                        );
                    }

                    await loadDocuments();
                } catch (error) {
                    alert(`Delete failed: ${error.message}`);
                    deleteButton.disabled = false;
                    deleteButton.textContent = "Delete";
                }
            });

            item.appendChild(name);
            item.appendChild(deleteButton);
            list.appendChild(item);
        });
    } catch (error) {
        list.textContent = `Could not load documents: ${error.message}`;
        console.error("loadDocuments error:", error);
    }
}

async function loadHistory() {
    const list = document.getElementById("historyList");

    if (!list) {
        console.error("historyList element not found");
        return;
    }

    list.textContent = "Loading...";

    try {
        const response = await fetch(`${API}/rag/history`, {
            method: "GET",
            headers: authHeaders()
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Could not load chat history");
        }

        const history = Array.isArray(data)
            ? data
            : (data.history || []);

        list.replaceChildren();

        if (history.length === 0) {
            list.textContent = "No chat history found.";
            return;
        }

        history.forEach(chat => {
            const item = document.createElement("div");
            item.className = "history-item";

            const question = document.createElement("p");
            question.textContent = `Question: ${chat.query || ""}`;

            const answer = document.createElement("p");
            answer.textContent = `Answer: ${chat.answer || ""}`;

            const date = document.createElement("small");
            date.textContent = chat.created_at
                ? `Date: ${chat.created_at}`
                : "";

            item.append(question, answer, date);
            list.appendChild(item);
        });

    } catch (error) {
        list.textContent = `Could not load history: ${error.message}`;
        console.error("loadHistory error:", error);
    }
}

// ================================
// RESTORE LOGIN AFTER PAGE REFRESH
// ================================

document.addEventListener("DOMContentLoaded", async function () {
    const token = localStorage.getItem("access_token");

    if (!token) {
        showLogin();
        return;
    }

    try {
        const response = await fetch(`${API}/auth/me`, {
            method: "GET",
            headers: authHeaders()
        });

        if (!response.ok) {
            logout();
            return;
        }

        const user = await response.json();

        document.getElementById("currentUser").textContent =
            user.name || user.email || localStorage.getItem("username") || "";

        localStorage.setItem("username", user.name || user.email || "");

        showDashboard();
        showSection("askSection");

    } catch (error) {
        console.error("Session restore failed:", error);
        showLogin();
    }
});
