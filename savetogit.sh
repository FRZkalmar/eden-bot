#!/bin/bash
# -------------------------
# Mac/Linux Git auto-save script
# -------------------------

# Check if there are changes
if git diff-index --quiet HEAD --; then
    echo "No changes to commit."
    exit 0
fi

# Ask for commit message
echo "Enter commit message (leave blank to use timestamp):"
read commit_message

# If blank, use timestamp in format "Feb 25 2026 10:24"
if [ -z "$commit_message" ]; then
    commit_message=$(date '+%b %d %Y %H:%M')
fi

# Add all changes
git add .

# Commit
git commit -m "$commit_message"

# Push to main branch
git push origin main

echo "✅ All changes pushed to GitHub!"