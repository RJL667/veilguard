#!/bin/sh
# Patch LibreChat API to include conversationId in Anthropic metadata
# Applied at Docker build time

FILE="/app/packages/api/dist/index.js"

echo "[PATCH] Patching $FILE..."

# Patch 1: Add conversation_id to metadata object
# Stock:   user_id: mergedOptions.user,
# Patched: user_id: mergedOptions.user, conversation_id: mergedOptions.conversationId,
sed -i 's|user_id: mergedOptions.user,|user_id: mergedOptions.user, conversation_id: mergedOptions.conversationId,|' "$FILE"
COUNT1=$(grep -c 'conversation_id: mergedOptions.conversationId' "$FILE")
echo "[PATCH] Metadata patch: $COUNT1 match(es)"

# Patch 2: Pass conversationId from req.body into modelOptions
# The compiled TS uses: { user: (_e = req.user) === null ... }
# We use a Python script for precise multi-line patching
python3 -c "
import re

with open('$FILE', 'r') as f:
    content = f.read()

# Find the Anthropic initializeAnthropic function's modelOptions
# Pattern: modelOptions: Object.assign(... { user: <null_check_for_req.user.id> })
# We need to add: , conversationId: req.body && req.body.conversationId
# after the user field assignment

# Match: user: (_X = req.user) === null || _X === void 0 ? void 0 : _X.id })
# Only in the Anthropic initialize block (near ANTHROPIC_REVERSE_PROXY)
pattern = r'(ANTHROPIC_REVERSE_PROXY.*?user: \(_\w+ = req\.user\) === null \|\| _\w+ === void 0 \? void 0 : _\w+\.id)([ }])'
match = re.search(pattern, content, re.DOTALL)
if match:
    # Insert conversationId after the user field
    old = match.group(0)
    new = match.group(1) + ', conversationId: req.body && req.body.conversationId' + match.group(2)
    content = content.replace(old, new, 1)
    with open('$FILE', 'w') as f:
        f.write(content)
    print('[PATCH] modelOptions patch applied successfully')
else:
    print('[PATCH] WARNING: modelOptions pattern not found')
"

# Verify
COUNT2=$(grep -c 'conversationId: req.body' "$FILE")
echo "[PATCH] ModelOptions patch: $COUNT2 match(es)"
echo "[PATCH] Done"
