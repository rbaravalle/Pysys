//|||||||||||||||||||||||||||||||||||||||||||||||

#include "GameState.hpp"
#include "ReloadMaterial.hpp"
#include "OgreWireBoundingBox.h" 

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

        tmk = 15.0;
        tmk2 = 25.0;
        mintm = 0.2;
        shininess = 1.0;
        steps = 140.0;
        ucolor = Vector3(0.8,0.7,0.6);
        ambient = 0.3;
        backIllum = 0.0;
        shadeCoeff = 1.1;
        specCoeff= 5.0;
        specMult = 0.1;
        misc = 1.3;
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
        input.open("media/fields/mengel3d.field");
        // input.open("media/fields/imagen3-1.field");
        if (!input.good()) {
                std::cout << "Unable to open field file!\n";
                exit();
        }
        int texW, texH, texD;
        input >> texW >> texH >> texD;

        // std::cout << "texD_p2: "  << texD_p2 << std::endl;
        // std::cout << "levels: "  << levels << std::endl;

        int* field = new int[texW * texH * texD];
        
        // texW = texH = texD = 256;

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
                // 0,                // number of mipmaps
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
                                idx = idx % (250*250*250);
                                *pDest++ = field[idx]; 
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

                                        *pDest++ =  levelField[idx]; 
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
        ////////////////////// Volume texture
        createVolumeTexture();

        //////////// Background color
        Ogre::Viewport* vp = OgreFramework::getSingletonPtr()->m_pViewport;
        vp->setBackgroundColour (ColourValue(0.1,0.1,0.1));

        //////////// Light
        m_pSceneMgr->setAmbientLight(ColourValue(0.1,0.1,0.1));

        light = m_pSceneMgr->createLight("Light");
        light->setType(Light::LT_POINT);
        light->setPosition(100,100,100);
        // light->setDirection(100,-100,100);
        light->setDiffuseColour(1,1,1);
        light->setSpecularColour(1.0,1.0,1.0);
        light->setSpotlightRange(Radian(M_PI/2), Radian(M_PI/3));
        // light->setAttenuation(20, 0.5, 1, 1);
        
        //////////// Shadows
        // m_pSceneMgr->setShadowTechnique(SHADOWTYPE_STENCIL_ADDITIVE);
        // m_pSceneMgr->setShadowTextureSettings( 256, 2);
        // m_pSceneMgr->setShadowTextureConfig( 0, 512, 512, PF_FLOAT16_R, 50 );

        ////////////////////// BREAD
        breadEntity = m_pSceneMgr->createEntity("BreadEntity", "Cube01.mesh");
        breadNode = m_pSceneMgr->getRootSceneNode()->createChildSceneNode("BreadNode");
        breadNode->attachObject(breadEntity);
        breadNode->setOrientation(Quaternion::IDENTITY);
        breadNode->setPosition(Vector3(0, 0, 0));
        breadNode->setScale(Vector3(10,10,10));
        breadEntity->setRenderQueueGroup(RENDER_QUEUE_8);

        breadEntity->getSubEntity(0)->setMaterialName("Bread","General");

        breadMat = breadEntity->getSubEntity(0)->getMaterial();
        

        /////////////////////// TABLE
        Entity* planeEntity;
        SceneNode* planeNode;
        planeEntity = m_pSceneMgr->createEntity("PlaneEntity", "Plane.mesh");
        planeEntity->getSubEntity(0)->setMaterialName("Table","General");
        planeNode = m_pSceneMgr->getRootSceneNode()->createChildSceneNode("PlaneNode");
        planeNode->attachObject(planeEntity);
        planeNode->setOrientation(Quaternion::IDENTITY);
        planeNode->setPosition(Vector3(0, -10, 0));
        planeNode->setScale(Vector3(10,10,10));

        /////////////////////// KNIFE
        Entity* knifeEntity;
        SceneNode* knifeNode;
        knifeEntity = m_pSceneMgr->createEntity("KnifeEntity", "knife.mesh");
        knifeEntity->getSubEntity(0)->setMaterialName("Knife","General");
        knifeNode = m_pSceneMgr->getRootSceneNode()->createChildSceneNode("KnifeNode");
        knifeNode->attachObject(knifeEntity);
        Quaternion ori(Radian(-0.5), Vector3(0,1,0));
        knifeNode->setOrientation(ori);
        knifeNode->setPosition(Vector3(30, -9, -30));
        knifeNode->setScale(Vector3(50,50,50));

        ////////////////////// BACKGROUND
        // // Create background material
        // MaterialPtr material = MaterialManager::getSingleton().create("Background", "General");
        // material->getTechnique(0)->getPass(0)->createTextureUnitState("steel.jpg");
        // material->getTechnique(0)->getPass(0)->setDepthCheckEnabled(false);
        // material->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
        // material->getTechnique(0)->getPass(0)->setLightingEnabled(false);
 
        // Create background rectangle covering the whole screen
        Rectangle2D* rect = new Rectangle2D(true);
        rect->setCorners(-1.0, 1.0, 1.0, -1.0);
        rect->setMaterial("Degrade");
 
        // Render the background before everything else
        rect->setRenderQueueGroup(RENDER_QUEUE_BACKGROUND);
 
        // Use infinite AAB to always stay visible
        AxisAlignedBox aabInf;
        aabInf.setInfinite();
        rect->setBoundingBox(aabInf);
 
        // Attach background to the scene
        SceneNode* node = m_pSceneMgr->getRootSceneNode()->createChildSceneNode("Background");
        node->attachObject(rect);


        /////////////// Light obj
        // Create background rectangle covering the whole screen
        lightEntity = m_pSceneMgr->createEntity("LightEntity", "Cube01.mesh");
        lightNode = m_pSceneMgr->getRootSceneNode()->createChildSceneNode("Light");
        lightEntity->getSubEntity(0)->setMaterialName("White","General");
        lightNode->attachObject(lightEntity);
        lightNode->setPosition(light->getPosition());
        lightNode->showBoundingBox(true);

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

                if(OgreFramework::getSingletonPtr()->m_pKeyboard->isKeyDown(OIS::KC_R))
                {
                        steps = 20;
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
                        OgreFramework::getSingletonPtr()->m_pTrayMgr->moveWidgetToTray(m_pDetailsPanel, OgreBites::TL_BOTTOMRIGHT, 0);
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
                // onLeftPressed(evt);
                // m_bLMouseDown = true;
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
        updateSliders();
        updateLight(timeSinceLastFrame);
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

        // Ogre::Vector3 campos = m_pCamera->getPosition();
        // try { fparams->setNamedConstant("uCamPos", campos); } catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uTMK", tmk); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uTMK2", tmk2); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uMinTm", mintm); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uShadeCoeff", shadeCoeff); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uSpecCoeff", specCoeff); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uSpecMult", specMult); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uShininess", shininess); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uMaxSteps", steps); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uAmbient", ambient); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uBackIllum", backIllum); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uMisc", misc); } 
        catch (Ogre::Exception) {}

}

void GameState::updateSliders()
{
        tmkSlider->setValue(tmk, false);
        tmk2Slider->setValue(tmk2, false);
        mintmSlider->setValue(mintm, false);
        shadeCoeffSlider->setValue(shadeCoeff, false);
        specCoeffSlider->setValue(specCoeff, false);
        specMultSlider->setValue(specMult, false);
        shininessSlider->setValue(shininess, false);
        stepsSlider->setValue(steps, false);
        ambientSlider->setValue(ambient, false);
        backIllumSlider->setValue(backIllum, false);
        miscSlider->setValue(misc, false);
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void GameState::updateLight(double timeSinceLastFrame)
{
        static double elapsed = 0;
        elapsed += timeSinceLastFrame * 0.001;
        double se = sin(elapsed);
        double ce = cos(elapsed);

        Vector3 pos = Vector3(100 * se, 100 , 100 * ce); 
        light->setPosition(pos);
        lightNode->setPosition(pos);
        // light->setDirection(-pos);
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

        m_pDetailsPanel = trayMgr->createParamsPanel(OgreBites::TL_NONE, 
                                                     "DetailsPanel", 200, items);
        m_pDetailsPanel->hide();

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

        tmkSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "tmk", 
                                              "tmk", 200,80,44,0,25,101);

        tmk2Slider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "tmk2", 
                                               "tmk2", 200,80,44,0,25,101);

        mintmSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "minTm", 
                                                "minTm", 200,80,44,0,1,101);

        shininessSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "shininess", 
                                                    "shininess",  200,80,44,0,10,101);

        stepsSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "steps", 
                                                "steps",  200,80,44,16,256,241);

        ambientSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "ambient", 
                                                  "ambient",  200,80,44,-3,3,61);

        backIllumSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "backIllum", 
                                          "back illumination", 200,80,44,0,3,31);

        shadeCoeffSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "shadeCoeff", 
                                                     "shadeCoeff", 200,80,44,0.1,5,50);

        specCoeffSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "specCoeff", 
                                                     "specCoeff", 200,80,44,0.1,5,50);

        specMultSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "specMult", 
                                                     "specMult", 200,80,44,0.1,2,39);

        miscSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "misc", 
                                               "misc", 200,80,44,0,10,101);

        // OgreBites::Button* reloadMaterialButton = 
        //         trayMgr->createButton(OgreBites::TL_RIGHT, 
        //                               "ReloadMaterial", 
        //                               "Reload material", 60);

        updateSliders();
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

        if (slider->getName() == "tmk") 
        {
                tmk = value;
        }

        if (slider->getName() == "tmk2") 
        {
                tmk2 = value;
        }

        if (slider->getName() == "minTm") 
        {
                mintm = value;
        }

        if (slider->getName() == "shadeCoeff") 
        {
                shadeCoeff = value;
        }

        if (slider->getName() == "specCoeff") 
        {
                specCoeff = value;
        }

        if (slider->getName() == "specMult") 
        {
                specMult = value;
        }

        if (slider->getName() == "shininess") 
        {
                shininess = value;
        }

        if (slider->getName() == "steps") 
        {
                steps = value;
        }

        if (slider == ambientSlider) 
        {
                ambient = value;
        }

        if (slider == backIllumSlider) 
        {
                backIllum = value;
        }

        if (slider == miscSlider) 
        {
                misc = value;
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
