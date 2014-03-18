cmake_minimum_required(VERSION 2.6)

include_directories("include")

include_directories("/usr/include")
include_directories("/usr/include/OIS")
include_directories("/usr/local/include")
include_directories("/usr/local/include/OGRE")
include_directories("/usr/local/include/OGRE/Overlay")

add_library(OgreFramework SHARED
  src/AdvancedOgreFramework.cpp
  src/AppStateManager.cpp
  src/DotSceneLoader.cpp
  src/GameState.cpp
  src/MenuState.cpp
  src/PauseState.cpp
)
target_link_libraries(OgreFramework OgreMain OgreTerrain OgreOverlay OIS)

add_executable(Demo
  src/DemoApp.cpp
  src/main.cpp
)
target_link_libraries(Demo OgreFramework)

install (TARGETS Demo DESTINATION ./)
install (TARGETS OgreFramework DESTINATION ./)

