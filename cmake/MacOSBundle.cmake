# MacOSBundle.cmake - Create Xenia.app bundle structure
#
# Call this from the root CMakeLists.txt after building xenia-app:
#   include(cmake/MacOSBundle.cmake)

if(APPLE)
  set(XENIA_APP_BUNDLE_NAME "Xenia")
  find_library(XENIA_SDL2_DYLIB
    NAMES SDL2
    PATHS /opt/homebrew/opt/sdl2/lib /opt/homebrew/lib
    REQUIRED
  )
  find_library(XENIA_VULKAN_LOADER_DYLIB
    NAMES vulkan.1 vulkan
    PATHS /opt/homebrew/opt/vulkan-loader/lib /opt/homebrew/lib
    REQUIRED
  )
  find_library(XENIA_MOLTENVK_DYLIB
    NAMES MoltenVK
    PATHS /opt/homebrew/opt/molten-vk/lib /opt/homebrew/lib
    REQUIRED
  )

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

  add_custom_command(TARGET xenia-app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Frameworks"
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Resources/vulkan/icd.d"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${XENIA_SDL2_DYLIB}"
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Frameworks/libSDL2-2.0.0.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${XENIA_VULKAN_LOADER_DYLIB}"
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Frameworks/libvulkan.1.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${XENIA_VULKAN_LOADER_DYLIB}"
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Frameworks/libvulkan.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${XENIA_MOLTENVK_DYLIB}"
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Frameworks/libMoltenVK.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PROJECT_SOURCE_DIR}/packaging/macos/MoltenVK_icd.json"
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Resources/vulkan/icd.d/MoltenVK_icd.json"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PROJECT_SOURCE_DIR}/LICENSE"
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Resources/LICENSE"
    COMMAND ${CMAKE_COMMAND} -E rm -f
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/MacOS/xenia-app"
    COMMAND /usr/bin/install_name_tool -id @rpath/libSDL2-2.0.0.dylib
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Frameworks/libSDL2-2.0.0.dylib"
    COMMAND /usr/bin/install_name_tool -id @rpath/libvulkan.1.dylib
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Frameworks/libvulkan.1.dylib"
    COMMAND /usr/bin/install_name_tool -id @rpath/libvulkan.dylib
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Frameworks/libvulkan.dylib"
    COMMAND /usr/bin/install_name_tool -id @rpath/libMoltenVK.dylib
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/Frameworks/libMoltenVK.dylib"
    COMMAND /usr/bin/install_name_tool -change /opt/homebrew/opt/sdl2/lib/libSDL2-2.0.0.dylib
      @rpath/libSDL2-2.0.0.dylib
      "$<TARGET_BUNDLE_DIR:xenia-app>/Contents/MacOS/${XENIA_APP_BUNDLE_NAME}"
    VERBATIM
  )

  # Set RPATH for the bundle so it finds dylibs in Frameworks/
  set_target_properties(xenia-app PROPERTIES
    INSTALL_RPATH "@executable_path/../Frameworks"
    BUILD_WITH_INSTALL_RPATH TRUE
  )

  # Custom target for creating DMG — also handles bundle setup
  add_custom_target(xenia-dmg
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
