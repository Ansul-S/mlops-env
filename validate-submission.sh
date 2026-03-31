#!/bin/bash

URL=$1

echo "🔍 Testing /reset endpoint..."
curl -s -X POST "$URL/reset" -H "Content-Type: application/json" -d '{}' > /dev/null
if [ $? -ne 0 ]; then
  echo "❌ /reset failed"
  exit 1
fi

echo "🔍 Testing /step endpoint..."
curl -s -X POST "$URL/step" -H "Content-Type: application/json" \
-d '{"action": {"action_type": "accept_record", "params": {}}}' > /dev/null

if [ $? -ne 0 ]; then
  echo "❌ /step failed"
  exit 1
fi

echo "✅ Basic validation passed!"
