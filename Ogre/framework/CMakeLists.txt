cmake_minimum_required(VERSION 2.6)

include_directories("include")

include_directories("/usr/include")
include_directories("/usr/include/OIS")
include_directories("/usr/include")
include_directories("/usr/include/OGRE")
include_directories("/usr/include/OGRE/Overlay")

add_library(OgreFramework SHARED
  src/AdvancedOgreFramework.cpp
  src/AppStateManager.cpp
  src/DotSceneLoader.cpp
  src/GameState.cpp
  src/MenuState.cpp
  src/PauseState.cpp
  src/ReloadMaterial.cpp
)

target_link_libraries(OgreFramework OgreMain OgreTerrain OgreOverlay OIS)
install (TARGETS OgreFramework DESTINATION ./)

