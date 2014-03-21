//|||||||||||||||||||||||||||||||||||||||||||||||

#include "GameState.hpp"
#include "ReloadMaterial.hpp"

//|||||||||||||||||||||||||||||||||||||||||||||||

using namespace Ogre;

GameState::GameState()
{
        m_MoveSpeed                 = 0.1f;
        m_RotateSpeed               = 0.3f;

        m_bLMouseDown       = false;
        m_bRMouseDown       = false;
        m_bQuit             = false;
        m_bSettingsMode     = false;

        m_pDetailsPanel             = 0;

        utmk = 4.0;
        utmk2 = 5.0;
        shininess = 1.0;
        steps = 64.0;
        ucolor = Vector3(1.0,1.0,1.0);
        ambient = 0;
        backIllum = 0;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::enter()
{
        OgreFramework::getSingletonPtr()->m_pLog->logMessage("Entering GameState...");

        m_pSceneMgr = OgreFramework::getSingletonPtr()->m_pRoot->createSceneManager(ST_GENERIC, "GameSceneMgr");
        m_pSceneMgr->setAmbientLight(Ogre::ColourValue(0.7f, 0.7f, 0.7f));

        m_pSceneMgr->addRenderQueueListener(OgreFramework::getSingletonPtr()->m_pOverlaySystem);

        m_pRSQ = m_pSceneMgr->createRayQuery(Ray());
        m_pRSQ->setQueryMask(OGRE_HEAD_MASK);

        m_pCamera = m_pSceneMgr->createCamera("GameCamera");
        m_pCamera->setPosition(Vector3(5, 60, 60));
        m_pCamera->lookAt(Vector3(5, 20, 0));
        m_pCamera->setNearClipDistance(5);

        m_pCamera->setAspectRatio(Real(OgreFramework::getSingletonPtr()->m_pViewport->getActualWidth()) /
                                  Real(OgreFramework::getSingletonPtr()->m_pViewport->getActualHeight()));

        OgreFramework::getSingletonPtr()->m_pViewport->setCamera(m_pCamera);
        m_pCurrentObject = 0;

        buildGUI();

        createScene();
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool GameState::pause()
{
        OgreFramework::getSingletonPtr()->m_pLog->logMessage("Pausing GameState...");

        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::resume()
{
        OgreFramework::getSingletonPtr()->m_pLog->logMessage("Resuming GameState...");

        buildGUI();

        OgreFramework::getSingletonPtr()->m_pViewport->setCamera(m_pCamera);
        m_bQuit = false;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::exit()
{
        shutdown();
        return;

        // OgreFramework::getSingletonPtr()->m_pLog->logMessage("Leaving GameState...");

        // m_pSceneMgr->destroyCamera(m_pCamera);
        // m_pSceneMgr->destroyQuery(m_pRSQ);
        // if(m_pSceneMgr)
        //         OgreFramework::getSingletonPtr()->m_pRoot->destroySceneManager(m_pSceneMgr)
                        ;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

static size_t getParentIndex(int x, int y, int z,
                             size_t fieldW, size_t fieldH, size_t fieldD) 
{
        size_t X = std::min(std::max(x*2, 0), (int)fieldW);
        size_t Y = std::min(std::max(y*2, 0), (int)fieldH);
        size_t Z = std::min(std::max(z*2, 0), (int)fieldD);

        return X + Y * fieldW + Z * fieldW * fieldH;
}
                                

static int sampleParentField(int* field, 
                             size_t fieldW, size_t fieldH, size_t fieldD,
                             int x, int y, int z) 
{
        const int width = 3;
        const int span = 1;
        float multipliers[width] = {0.2, 0.6, 0.2};

        float* mult = multipliers + span;
        int res = 0;
        

        for (int dx = -span; dx <= span; dx++) {
                for (int dy = -span; dy <= span; dy++) {
                      for (int dz = -span; dz <= span; dz++) {
                                size_t idx = getParentIndex(x+dx,y+dy,z+dz,
                                                            fieldW,fieldH,fieldD);
                                res += field[idx] * mult[dx] * mult[dy] * mult[dz];
                                
                        }
                }
        }

        return res;

}

void GameState::createVolumeTexture()
{
        std::ifstream input;
        input.open("media/fields/3Dbread.field");
        int texW, texH, texD;
        input >> texW >> texH >> texD;

        // std::cout << "texD_p2: "  << texD_p2 << std::endl;
        // std::cout << "levels: "  << levels << std::endl;

        int* field = new int[texW * texH * texD];

        int i = 0;
        while (!input.eof() && i < texW * texH * texD) {
                input >> field[i++];
        }

        TextureManager::getSingleton().setDefaultNumMipmaps(MIP_UNLIMITED);

        breadTex = TextureManager::getSingleton().createManual(
                "volumeTex", // name
                ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, // Group
                TEX_TYPE_3D,      // type
                texW, texH, texD,    // width height depth
                MIP_UNLIMITED,                // number of mipmaps
                PF_L8,     // pixel format -> 8 bits luminance
                TU_DEFAULT);      // usage; should be TU_DYNAMIC_WRITE_ONLY_DISCARDABLE for
        // textures updated very often (e.g. each frame)

        breadTex->createInternalResources();

        ////////////// Set initial texture level (0)
        // Get the pixel buffer
        HardwarePixelBufferSharedPtr pixelBuffer = breadTex->getBuffer();

        // Lock the pixel buffer and get a pixel box (for performance use HBL_DISCARD!)
        pixelBuffer->lock(HardwareBuffer::HBL_NORMAL); 

        const PixelBox& pixelBox = pixelBuffer->getCurrentLock();
 
        uint8* pDest = static_cast<uint8*>(pixelBox.data);
        int colBytes = Ogre::PixelUtil::getNumElemBytes(pixelBox.format);
 
        // Fill in some pixel data. This will give a semi-transparent blue,
        // but this is of course dependent on the chosen pixel format.
        for (size_t z = 0; z < texD; z++)
        {
                for(size_t y = 0; y < texH; y++)
                {
                        for(size_t x = 0; x < texW; x++)
                        {
                                int idx = x + y * texW + z * texW * texH;
                                *pDest++ = 255 - field[idx]; 
                        }
                        pDest += pixelBox.getRowSkip() * colBytes;
                }
                pDest += pixelBox.getSliceSkip() * colBytes;
        }
 
        // Unlock the pixel buffer
        pixelBuffer->unlock();


        int levels = breadTex->getNumMipmaps();
        std::cout << "breadTex mip levels: " << levels << std::endl;
        int level = 1;

        int** levelFields = new int*[levels];

        levelFields[0] = field;

        int levelTexW = texW;
        int levelTexH = texH;
        int levelTexD = texD;

        /// This assumes NPOT textures are possible

        for (;level < levels; ++level) {

                int lastTexW = levelTexW;
                int lastTexH = levelTexH;
                int lastTexD = levelTexD;

                levelTexW = std::max(1.0, floor(texW / pow(2,level)));
                levelTexH = std::max(1.0, floor(texH / pow(2,level)));
                levelTexD = std::max(1.0, floor(texD / pow(2,level)));

                int* levelField = new int[levelTexW * levelTexH * levelTexD];

                levelFields[level] = levelField;

                int* parentField = levelFields[level-1];

                // Get the pixel buffer
                HardwarePixelBufferSharedPtr pixelBuffer = breadTex->getBuffer(0,level);

                // Lock the pixel buffer and get a pixel box 
                pixelBuffer->lock(HardwareBuffer::HBL_NORMAL); 

                const PixelBox& pixelBox = pixelBuffer->getCurrentLock();
 
                uint8* pDest = static_cast<uint8*>(pixelBox.data);
                int colBytes = Ogre::PixelUtil::getNumElemBytes(pixelBox.format);
 
                // Fill in some pixel data. This will give a semi-transparent blue,
                // but this is of course dependent on the chosen pixel format.
                for (size_t z = 0; z < levelTexD; z++)
                {
                        for(size_t y = 0; y < levelTexH; y++)
                        {
                                for(size_t x = 0; x < levelTexW; x++)
                                {

                                        int idx = x+y*levelTexW+z*levelTexW*levelTexH;

                                        levelField[idx] = 
                                                sampleParentField(parentField,
                                                                  lastTexW,
                                                                  lastTexH,
                                                                  lastTexD,
                                                                  x,y,z);

                                        *pDest++ = 255 - levelField[idx]; 
                                }
                                pDest += pixelBox.getRowSkip() * colBytes;
                        }
                        pDest += pixelBox.getSliceSkip() * colBytes;
                }
 
                // Unlock the pixel buffer
                pixelBuffer->unlock();
        }

        breadTex->load();

        // for (int i = 0 ; i < levels; ++i) 
        //         delete[] levelFields[i];


}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::createScene()
{
        createVolumeTexture();

        Ogre::Viewport* vp = OgreFramework::getSingletonPtr()->m_pViewport;
        vp->setBackgroundColour (ColourValue(0.1,0.2,0.1));

        m_pSceneMgr->createLight("Light")->setPosition(75,75,75);

        breadEntity = m_pSceneMgr->createEntity("BreadEntity", "Cube01.mesh");
        breadNode = m_pSceneMgr->getRootSceneNode()->createChildSceneNode("BreadNode");
        breadNode->attachObject(breadEntity);
        breadNode->setPosition(Vector3(0, 0, 0));
        breadNode->setScale(Vector3(10,10,10));

        breadEntity->getSubEntity(0)->setMaterialName("Bread","General");

        breadMat = breadEntity->getSubEntity(0)->getMaterial();
        
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool GameState::keyPressed(const OIS::KeyEvent &keyEventRef)
{
        if(m_bSettingsMode == true)
        {
                if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_S))
                {
                        OgreBites::SelectMenu* pMenu = (OgreBites::SelectMenu*)OgreFramework::getSingletonPtr()->m_pTrayMgr->getWidget("DisplayModeSelMenu");
                        if(pMenu->getSelectionIndex() + 1 < (int)pMenu->getNumItems())
                                pMenu->selectItem(pMenu->getSelectionIndex() + 1);
                }

                if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_W))
                {
                        OgreBites::SelectMenu* pMenu = (OgreBites::SelectMenu*)OgreFramework::getSingletonPtr()->m_pTrayMgr->getWidget("DisplayModeSelMenu");
                        if(pMenu->getSelectionIndex() - 1 >= 0)
                                pMenu->selectItem(pMenu->getSelectionIndex() - 1);
                }
        }

        if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_ESCAPE))
        {
                // pushAppState(findByName("PauseState"));
                exit();
                return true;
        }

        if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_I))
        {
                if(m_pDetailsPanel->getTrayLocation() == OgreBites::TL_NONE)
                {
                        OgreFramework::getSingletonPtr()->m_pTrayMgr->moveWidgetToTray(m_pDetailsPanel, OgreBites::TL_LEFT, 0);
                        m_pDetailsPanel->show();
                }
                else
                {
                        OgreFramework::getSingletonPtr()->m_pTrayMgr->removeWidgetFromTray(m_pDetailsPanel);
                        m_pDetailsPanel->hide();
                }
        }

        if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_TAB))
        {
                m_bSettingsMode = !m_bSettingsMode;
                return true;
        }

        if(m_bSettingsMode && OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_RETURN) ||
           OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_NUMPADENTER))
        {
        }

        if(!m_bSettingsMode || (m_bSettingsMode && !OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_O)))
                OgreFramework::getSingletonPtr()->keyPressed(keyEventRef);

        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool GameState::keyReleased(const OIS::KeyEvent &keyEventRef)
{
        OgreFramework::getSingletonPtr()->keyPressed(keyEventRef);
        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool GameState::mouseMoved(const OIS::MouseEvent &evt)
{
        if(OgreFramework::getSingletonPtr()->m_pTrayMgr->injectMouseMove(evt)) return true;

        if(m_bRMouseDown)
        {
                m_pCamera->yaw(Degree(evt.state.X.rel * -0.1f));
                m_pCamera->pitch(Degree(evt.state.Y.rel * -0.1f));
        }

        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool GameState::mousePressed(const OIS::MouseEvent &evt, OIS::MouseButtonID id)
{
        if(OgreFramework::getSingletonPtr()->m_pTrayMgr->injectMouseDown(evt, id)) return true;

        if(id == OIS::MB_Left)
        {
                onLeftPressed(evt);
                m_bLMouseDown = true;
        }
        else if(id == OIS::MB_Right)
        {
                m_bRMouseDown = true;
        }

        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool GameState::mouseReleased(const OIS::MouseEvent &evt, OIS::MouseButtonID id)
{
        if(OgreFramework::getSingletonPtr()->m_pTrayMgr->injectMouseUp(evt, id)) return true;

        if(id == OIS::MB_Left)
        {
                m_bLMouseDown = false;
        }
        else if(id == OIS::MB_Right)
        {
                m_bRMouseDown = false;
        }

        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::onLeftPressed(const OIS::MouseEvent &evt)
{
        if(m_pCurrentObject)
        {
                m_pCurrentObject->showBoundingBox(false);
                m_pCurrentEntity->getSubEntity(0)->setMaterial(breadMat);
        }

        Ogre::Ray mouseRay = m_pCamera->getCameraToViewportRay(OgreFramework::getSingletonPtr()->m_pMouse->getMouseState().X.abs / float(evt.state.width),
                                                               OgreFramework::getSingletonPtr()->m_pMouse->getMouseState().Y.abs / float(evt.state.height));
        m_pRSQ->setRay(mouseRay);
        m_pRSQ->setSortByDistance(true);

        Ogre::RaySceneQueryResult &result = m_pRSQ->execute();
        Ogre::RaySceneQueryResult::iterator itr;

        for(itr = result.begin(); itr != result.end(); itr++)
        {
                if(itr->movable)
                {
                        OgreFramework::getSingletonPtr()->m_pLog->logMessage("MovableName: " + itr->movable->getName());
                        m_pCurrentObject = m_pSceneMgr->getEntity(itr->movable->getName())->getParentSceneNode();
                        OgreFramework::getSingletonPtr()->m_pLog->logMessage("ObjName " + m_pCurrentObject->getName());
                        m_pCurrentObject->showBoundingBox(true);
                        m_pCurrentEntity = m_pSceneMgr->getEntity(itr->movable->getName());
                        break;
                }
        }
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::moveCamera()
{
        if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_LSHIFT))
                m_pCamera->moveRelative(m_TranslateVector);
        m_pCamera->moveRelative(m_TranslateVector / 10);
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::getInput()
{
        if(m_bSettingsMode == false)
        {
                if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_A))
                        m_TranslateVector.x = -m_MoveScale;

                if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_D))
                        m_TranslateVector.x = m_MoveScale;

                if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_W))
                        m_TranslateVector.z = -m_MoveScale;

                if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_S))
                        m_TranslateVector.z = m_MoveScale;
        }
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::update(double timeSinceLastFrame)
{
        m_FrameEvent.timeSinceLastFrame = timeSinceLastFrame;
        OgreFramework::getSingletonPtr()->m_pTrayMgr->frameRenderingQueued(m_FrameEvent);

        if(m_bQuit == true)
        {
                popAppState();
                return;
        }

        if(!OgreFramework::getSingletonPtr()->m_pTrayMgr->isDialogVisible()) {
                if (m_pDetailsPanel->isVisible()) {
                        Ogre::Vector3 pos    = m_pCamera->getDerivedPosition();
                        Ogre::Quaternion ori = m_pCamera->getDerivedOrientation();
                        m_pDetailsPanel->setParamValue(0, StringConverter::toString(pos.x));
                        m_pDetailsPanel->setParamValue(1, StringConverter::toString(pos.y));
                        m_pDetailsPanel->setParamValue(2, StringConverter::toString(pos.z));
                        m_pDetailsPanel->setParamValue(3, StringConverter::toString(ori.w));
                        m_pDetailsPanel->setParamValue(4, StringConverter::toString(ori.x));
                        m_pDetailsPanel->setParamValue(5, StringConverter::toString(ori.y));
                        m_pDetailsPanel->setParamValue(6, StringConverter::toString(ori.z));
                        if(m_bSettingsMode)
                                m_pDetailsPanel->setParamValue(7, "Buffered Input");
                        else
                                m_pDetailsPanel->setParamValue(7, "Un-Buffered Input");
                }
        }


        m_MoveScale = m_MoveSpeed   * timeSinceLastFrame;
        m_RotScale  = m_RotateSpeed * timeSinceLastFrame;

        m_TranslateVector = Vector3::ZERO;

        getInput();
        moveCamera();

        updateMaterial();
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::updateMaterial()
{
        Ogre::Pass* pass = breadMat->getBestTechnique()->getPass(0);

        Ogre::GpuProgramParametersSharedPtr fparams = pass->getFragmentProgramParameters();

        Ogre::Vector3 lightpos(1,0,0);

        static double ro = 0;
        ro += 0.005;
        
        // lightpos = Ogre::Vector3(sin(ro), cos(ro), 0);
        // try { fparams->setNamedConstant("uLightP", lightpos); } catch (Ogre::Exception) {}

        Ogre::Vector3 campos = m_pCamera->getPosition();
        try { fparams->setNamedConstant("uCamPos", campos); } catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uTMK", utmk); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uTMK2", utmk2); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uShininess", shininess); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uMaxSteps", steps); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uAmbient", ambient); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uBackIllum", backIllum); } 
        catch (Ogre::Exception) {}

}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::buildGUI()
{

        OgreBites::SdkTrayManager* trayMgr = OgreFramework::getSingletonPtr()->m_pTrayMgr;


        trayMgr->showFrameStats(OgreBites::TL_BOTTOMLEFT);
        // trayMgr->showLogo(OgreBites::TL_BOTTOMRIGHT);
        // trayMgr->createLabel(OgreBites::TL_TOP, "GameLbl", "Game mode", 250);
        trayMgr->showCursor();

        Ogre::StringVector items;
        items.push_back("cam.pX");
        items.push_back("cam.pY");
        items.push_back("cam.pZ");
        items.push_back("cam.oW");
        items.push_back("cam.oX");
        items.push_back("cam.oY");
        items.push_back("cam.oZ");
        items.push_back("Mode");

        m_pDetailsPanel = trayMgr->createParamsPanel(OgreBites::TL_LEFT, "DetailsPanel", 200, items);
        m_pDetailsPanel->show();

        // Ogre::String infoText = "Controls\n[TAB] - Switch input mode\n\n[W] - Forward / Mode up\n[S] - Backwards/ Mode down\n[A] - Left\n";
        // infoText.append("[D] - Right\n\nPress [SHIFT] to move faster\n\n[O] - Toggle FPS / logo\n");
        // infoText.append("[Print] - Take screenshot\n\n[ESC] - Exit");
        // trayMgr->createTextBox(OgreBites::TL_RIGHT, "InfoPanel", infoText, 300, 220);

        Ogre::StringVector displayModes;
        displayModes.push_back("Solid mode");
        displayModes.push_back("Wireframe mode");
        displayModes.push_back("Point mode");
        trayMgr->createLongSelectMenu(OgreBites::TL_TOPRIGHT, 
                                      "DisplayModeSelMenu", 
                                      "Display Mode", 200, 3, displayModes);

        OgreBites::Slider* utmkSlider = 
                trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "utmk", "utmk", 
                                          200,80,44,0,10,101);

        OgreBites::Slider* utmk2Slider = 
        trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "utmk2", "utmk2", 
                                  200,80,44,0,10,101);

        OgreBites::Slider* shininessSlider = 
        trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "shininess", "shininess",  
                                  200,80,44,0,10,101);

        OgreBites::Slider* stepsSlider = 
        trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "steps", "steps",  
                                  200,80,44,20,200,21);

        OgreBites::Slider* ambientSlider = 
        trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "ambient", "ambient",  
                                  200,80,44,0,3,31);

        OgreBites::Slider* backIllumSlider = 
                trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "backIllum", 
                                          "back illumination",
                                          200,80,44,0,3,31);


        // OgreBites::Button* reloadMaterialButton = 
        //         trayMgr->createButton(OgreBites::TL_RIGHT, 
        //                               "ReloadMaterial", 
        //                               "Reload material", 60);


        utmkSlider->setValue(utmk, false);
        utmk2Slider->setValue(utmk2, false);
        shininessSlider->setValue(shininess, false);
        stepsSlider->setValue(steps, false);
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::buttonHit(OgreBites::Button* button)
{
        if (button->getName() == "ReloadMaterial") {

                ReloadMaterial("Bread", 
                               "Bread",
                               "Bread.material",
                               true);

        }

}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::sliderMoved(OgreBites::Slider * slider)
{
        float value = slider->getValue();

        if (slider->getName() == "utmk") 
        {
                utmk = value;
        }

        if (slider->getName() == "utmk2") 
        {
                utmk2 = value;
        }

        if (slider->getName() == "shininess") 
        {
                shininess = value;
        }

        if (slider->getName() == "steps") 
        {
                steps = value;
        }

        if (slider->getName() == "ambient") 
        {
                ambient = value;
        }

        if (slider->getName() == "backIllum") 
        {
                backIllum = value;
        }

}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::itemSelected(OgreBites::SelectMenu* menu)
{
        switch(menu->getSelectionIndex())
        {
        case 0:
                m_pCamera->setPolygonMode(Ogre::PM_SOLID);break;
        case 1:
                m_pCamera->setPolygonMode(Ogre::PM_WIREFRAME);break;
        case 2:
                m_pCamera->setPolygonMode(Ogre::PM_POINTS);break;
        }
}

//|||||||||||||||||||||||||||||||||||||||||||||||