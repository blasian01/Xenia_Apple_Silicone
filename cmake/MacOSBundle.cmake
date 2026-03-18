# MacOSBundle.cmake - Create Xenia.app bundle structure
#
# Call this from the root CMakeLists.txt after building xenia-app:
#   include(cmake/MacOSBundle.cmake)

if(APPLE)
  set(XENIA_APP_BUNDLE_NAME "Xenia")

  # Set the target to be a macOS app bundle
  set_target_properties(xenia-app PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_INFO_PLIST "${PROJECT_SOURCE_DIR}/packaging/macos/Info.plist"
    MACOSX_BUNDLE_BUNDLE_NAME "${XENIA_APP_BUNDLE_NAME}"
    MACOSX_BUNDLE_GUI_IDENTIFIER "org.xenia.xenia-app"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0.0"
    MACOSX_BUNDLE_BUNDLE_VERSION "1"
    OUTPUT_NAME "${XENIA_APP_BUNDLE_NAME}"
  )

  # Set RPATH for the bundle so it finds dylibs in Frameworks/
  set_target_properties(xenia-app PROPERTIES
    INSTALL_RPATH "@executable_path/../Frameworks"
    BUILD_WITH_INSTALL_RPATH TRUE
  )

  # Custom target for creating DMG — also handles bundle setup
  add_custom_target(xenia-dmg
    # Create bundle directories
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Frameworks"
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Resources"
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Resources/vulkan/icd.d"

    # Copy license
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PROJECT_SOURCE_DIR}/LICENSE"
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Resources/LICENSE"

    # Create DMG
    COMMAND bash "${PROJECT_SOURCE_DIR}/packaging/macos/create_dmg.sh"
      "$<TARGET_BUNDLE_DIR:xenia-app>"
      "${PROJECT_SOURCE_DIR}/build/Xenia.dmg"

    DEPENDS xenia-app
    COMMENT "Packaging Xenia.app and creating DMG..."
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    VERBATIM
  )
endif()
