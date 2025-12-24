#!/usr/bin/env python3
import sys

with open('src/class_context_extractor.cpp', 'r') as f:
    content = f.read()

# Replace the problematic line
content = content.replace('#include <set>', '#include <set>')

with open('src/class_context_extractor.cpp', 'w') as f:
    f.write(content)
