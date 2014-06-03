cmake_minimum_required(VERSION 2.6)

project (OgreSys)

SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/output)
# SET(CMAKE_CXX_FLAGS "-O3 -g")
SET(CMAKE_CXX_FLAGS "-g")
SET(DESTINATION "output")

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
  src/RenderState.cpp
  src/ReloadMaterial.cpp
  src/Volume.cpp
  src/VolumeBoundingCubes.cpp
  src/RenderTex.cpp
)

target_link_libraries(OgreFramework OgreMain OgreTerrain OgreOverlay OIS)
install (TARGETS OgreFramework DESTINATION ./)

add_executable(OgreBread src/BreadApp.cpp src/main.cpp)

target_link_libraries(OgreBread OgreFramework OgreMain)

install (TARGETS OgreBread DESTINATION ./)
install (DIRECTORY ${PROJECT_SOURCE_DIR}/media  DESTINATION ./)
install (
  FILES 
  ${PROJECT_SOURCE_DIR}/config/plugins.cfg  
  ${PROJECT_SOURCE_DIR}/config/resources.cfg  
  DESTINATION ./)
