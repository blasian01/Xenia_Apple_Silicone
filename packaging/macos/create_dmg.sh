#!/bin/bash
# create_dmg.sh - Package Xenia.app into a distributable DMG
#
# Usage: ./create_dmg.sh [path/to/Xenia.app] [output.dmg]

set -e

APP_PATH="${1:-build-macos/Xenia.app}"
DMG_PATH="${2:-Xenia.dmg}"
DMG_VOLUME_NAME="Xenia - Xbox 360 Emulator"
DMG_SIZE="200m"

if [ ! -d "$APP_PATH" ]; then
    echo "Error: App bundle not found at $APP_PATH"
    echo "Build the project first with: cmake --build build-macos --target xenia-app"
    exit 1
fi

echo "Creating DMG from $APP_PATH..."

# Create a temporary directory for DMG contents
DMG_STAGING=$(mktemp -d)
cp -R "$APP_PATH" "$DMG_STAGING/"

# Create a symbolic link to /Applications for drag-and-drop install
ln -s /Applications "$DMG_STAGING/Applications"

# Remove any existing DMG
rm -f "$DMG_PATH"

# Create the DMG
hdiutil create \
    -volname "$DMG_VOLUME_NAME" \
    -srcfolder "$DMG_STAGING" \
    -ov \
    -format UDZO \
    -imagekey zlib-level=9 \
    "$DMG_PATH"

# Clean up
rm -rf "$DMG_STAGING"

echo ""
echo "✅ DMG created: $DMG_PATH"
echo "   Volume: $DMG_VOLUME_NAME"
echo ""
echo "To install: Open the DMG and drag Xenia.app to Applications."
