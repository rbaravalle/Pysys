cmake_minimum_required(VERSION 2.6)

project (OgreSys)

add_subdirectory(framework)

include_directories("include")
include_directories("framework/include")
include_directories("/usr/local/include/OGRE")
include_directories("/usr/local/include/OGRE/Overlay")
include_directories("/usr/include/OIS")

add_executable(OgreBread src/BreadApp.cpp src/main.cpp)

target_link_libraries(OgreBread OgreFramework OgreMain)

SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/output)

install (TARGETS OgreBread DESTINATION ./)
install (DIRECTORY ${PROJECT_SOURCE_DIR}/framework/media  DESTINATION ./)
install (
  FILES 
  ${PROJECT_SOURCE_DIR}/framework/config/plugins.cfg  
  ${PROJECT_SOURCE_DIR}/framework/config/resources.cfg  
  DESTINATION ./)
