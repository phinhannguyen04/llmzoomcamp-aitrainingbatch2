# llmzoomcamp-aitraining
# 📘 Assignment Submission Instructions (GitHub)

This course uses **GitHub** for submitting coding exercises.  
Follow the steps below carefully to ensure your work is graded correctly.

---

## 1️⃣ Before You Start

You must have:
- A **GitHub account**
- **Git** installed on your computer

---

## 2️⃣ Clone the Assignment Repository

Open your terminal and run:

```bash
git clone <repository-url>
cd <repository-name>
```
Do not work directly on the main branch.

## 3️⃣ Create Your Personal Submission Branch

Each student must create one personal branch for submission.

Branch Naming Format
```bash
<assignment-name>/<your-name>
```
Create the Branch
Run the following command from the repository root:
```bash
git checkout -b <assignment-name>/<your-name>
```

Example:
```bash
git checkout -b advance-rag/nguyen-tran
```
Branch Rules
- Work only on your own branch
- Do not rename your branch after creation
- Never push directly to main
- Do not work on other’s branch

## 4️⃣ Commit Your Work

Commit your work regularly with meaningful messages:
```bash
git add .
git commit -m "Implement feature X"
```
✅ Commit messages should clearly describe what you changed.

## 5️⃣ Push Your Branch to GitHub

Push your branch to GitHub:
```bash
git push origin <assignment-name>/<your-name>
```
## 6️⃣ Submit via Pull Request (Required)

- Open the assignment repository 
- Click Pull Requests → New Pull Request
- Select:
  - Base branch: ```main```
  - Compare branch: ```<assignment-name>/<your-name>```
- Title your Pull Request exactly as follows:
  - Assignment: ```<assignment-name> – <your-name>```
- Click Create Pull Request
📌 Only one Pull Request is allowed per assignment.

## 7️⃣ Submission Rules

The Pull Request creation time is considered your submission time
You may push updates before the deadline
Do not force-push or delete your branch

## ✅ Submission Checklist

Before the deadline, confirm:
- Code is on your branch
- All changes are committed and pushed
- Pull Request is open to main
- Code runs without errors
