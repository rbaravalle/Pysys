//|||||||||||||||||||||||||||||||||||||||||||||||

#include "GameState.hpp"

//|||||||||||||||||||||||||||||||||||||||||||||||

using namespace Ogre;

//|||||||||||||||||||||||||||||||||||||||||||||||

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
        steps = 128.0;

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
        OgreFramework::getSingletonPtr()->m_pLog->logMessage("Leaving GameState...");

        m_pSceneMgr->destroyCamera(m_pCamera);
        m_pSceneMgr->destroyQuery(m_pRSQ);
        if(m_pSceneMgr)
                OgreFramework::getSingletonPtr()->m_pRoot->destroySceneManager(m_pSceneMgr);
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::createVolumeTexture()
{
        std::ifstream input;
        input.open("media/fields/3Dbread.field");
        int texW, texH, texD;
        input >> texW >> texH >> texD;
        std::cout << "Width: " << texW;
        std::cout << " Height: " << texH;
        std::cout << " Depth: " << texD << std::endl;

        int* field = new int[texW * texH * texD];

        int i = 0;
        while (!input.eof() && i < texW * texH * texD) {
                input >> field[i++];
        }

        breadTex = TextureManager::getSingleton().createManual(
                "volumeTex", // name
                ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, // Group
                TEX_TYPE_3D,      // type
                texW, texH, texD,    // width height depth
                0,                // number of mipmaps
                PF_BYTE_BGRA,     // pixel format
                TU_DEFAULT);      // usage; should be TU_DYNAMIC_WRITE_ONLY_DISCARDABLE for
        // textures updated very often (e.g. each frame)


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
                                *pDest++ = x; // B
                                *pDest++ = y; // G
                                *pDest++ = 255 - field[x + y * texW + z * texW * texH]; // R
                                *pDest++ = 127; // A
                        }
                        pDest += pixelBox.getRowSkip() * colBytes;
                }
                pDest += pixelBox.getSliceSkip() * colBytes;
        }
 
        // Unlock the pixel buffer
        pixelBuffer->unlock();

}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::createScene()
{
        createVolumeTexture();

        Ogre::Viewport* vp = OgreFramework::getSingletonPtr()->m_pViewport;
        vp->setBackgroundColour (ColourValue(0.1,0.5,0.5));

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
                pushAppState(findByName("PauseState"));
                return true;
        }

        if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_I))
        {
                if(m_pDetailsPanel->getTrayLocation() == OgreBites::TL_NONE)
                {
                        OgreFramework::getSingletonPtr()->m_pTrayMgr->moveWidgetToTray(m_pDetailsPanel, OgreBites::TL_TOPLEFT, 0);
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

        m_pDetailsPanel = trayMgr->createParamsPanel(OgreBites::TL_TOPLEFT, "DetailsPanel", 200, items);
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
                trayMgr->createLongSlider(OgreBites::TL_LEFT, "utmk", "utmk", 
                                          200,80,44,0,10,101);

        OgreBites::Slider* utmk2Slider = 
        trayMgr->createLongSlider(OgreBites::TL_LEFT, "utmk2", "utmk2", 
                                  200,80,44,0,10,101);

        OgreBites::Slider* shininessSlider = 
        trayMgr->createLongSlider(OgreBites::TL_LEFT, "shininess", "shininess",  
                                  200,80,44,0,10,101);

        OgreBites::Slider* stepsSlider = 
        trayMgr->createLongSlider(OgreBites::TL_LEFT, "steps", "steps",  
                                  200,80,44,20,200,21);

        utmkSlider->setValue(utmk, false);
        utmk2Slider->setValue(utmk2, false);
        shininessSlider->setValue(shininess, false);
        stepsSlider->setValue(steps, false);
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
