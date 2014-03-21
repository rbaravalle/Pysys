#ifndef RELOAD_MATERIAL_HPP
#define RELOAD_MATERIAL_HPP

#include <Ogre.h>
#include <OgreString.h>
#include <OgreStringConverter.h>

void UnloadResource(Ogre::ResourceManager* resMgr, const std::string& resourceName);
void LoadResource(Ogre::ResourceManager* resMgr, const std::string& resourceName, 
                  const std::string& resourceGroup);
void UnloadMaterials(const std::string& filename);
void UnloadVertexPrograms(const std::string& filename);
void UnloadFragmentPrograms(const std::string& filename);
void ReloadMaterial(const std::string& materialName, const std::string& groupName, 
                    const std::string& filename, bool parseMaterialScript);


#endif // RELOAD_MATERIAL_HPP
