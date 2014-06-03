//|||||||||||||||||||||||||||||||||||||||||||||||

#ifndef GAME_STATE_HPP
#define GAME_STATE_HPP

//|||||||||||||||||||||||||||||||||||||||||||||||

#include "AppState.hpp"
#include "ReloadMaterial.hpp"
#include "DotSceneLoader.hpp"

#include <OgreSubEntity.h>
#include <OgreMaterialManager.h>

//|||||||||||||||||||||||||||||||||||||||||||||||

enum QueryFlags
{
        OGRE_HEAD_MASK  = 1<<0,
        CUBE_MASK       = 1<<1
};

//|||||||||||||||||||||||||||||||||||||||||||||||

class GameState : public AppState
{
public:
        GameState();

        DECLARE_APPSTATE_CLASS(GameState)

        void enter();
        void createScene();
        void exit();
        bool pause();
        void resume();

        void moveCamera();
        void getInput();
        void buildGUI();

        bool keyPressed(const OIS::KeyEvent &keyEventRef);
        bool keyReleased(const OIS::KeyEvent &keyEventRef);

        bool mouseMoved(const OIS::MouseEvent &arg);
        bool mousePressed(const OIS::MouseEvent &arg, OIS::MouseButtonID id);
        bool mouseReleased(const OIS::MouseEvent &arg, OIS::MouseButtonID id);

        void onLeftPressed(const OIS::MouseEvent &evt);
        void itemSelected(OgreBites::SelectMenu* menu);
	void sliderMoved(OgreBites::Slider * slider);
	void buttonHit(OgreBites::Button* button);

        void update(double timeSinceLastFrame);

private:

        void createVolumeTexture();
        void updateMaterial();
        void updateSliders();
        void updateLight(double timeSinceLastFrame);

        Ogre::SceneNode*                        breadNode;
        Ogre::Entity*                           breadEntity;
        Ogre::MaterialPtr                       breadMat; 
        Ogre::TexturePtr                        breadTex;

        Ogre::SceneNode*                        tableNode;
        Ogre::Entity*                           tableEntity;

        Ogre::SceneNode*                        knifeNode;
        Ogre::Entity*                           knifeEntity;

        Ogre::Light*                            light;
        Ogre::Entity*                           lightEntity;
        Ogre::SceneNode*                        lightNode;

        Ogre::SceneNode*                        backgroundNode;

        float tmk;
        float tmk2;
        float shininess;
        float steps;
        float ambient;
        float backIllum;
        float mintm;
        float shadeCoeff;
        float specCoeff;
        float specMult;
        float misc;
        Ogre::Vector3 ucolor;

        // Ogre::SceneNode*                        m_pOgreHeadNode;
        // Ogre::Entity*                           m_pOgreHeadEntity;
        // Ogre::MaterialPtr                       m_pOgreHeadMat;
        // Ogre::MaterialPtr                       m_pOgreHeadMatHigh;

        OgreBites::ParamsPanel*                 m_pDetailsPanel;
        bool                                    m_bQuit;

        Ogre::Vector3                           m_TranslateVector;
        Ogre::Real                              m_MoveSpeed;
        Ogre::Degree                            m_RotateSpeed;
        float                                   m_MoveScale;
        Ogre::Degree                            m_RotScale;

        Ogre::RaySceneQuery*                    m_pRSQ;
        Ogre::SceneNode*                        m_pCurrentObject;
        Ogre::Entity*                           m_pCurrentEntity;
        bool                                    m_bLMouseDown, m_bRMouseDown;
        bool                                    m_bSettingsMode;


        OgreBites::Slider* tmkSlider;
        OgreBites::Slider* tmk2Slider;
        OgreBites::Slider* mintmSlider;
        OgreBites::Slider* shininessSlider;
        OgreBites::Slider* stepsSlider;
        OgreBites::Slider* ambientSlider;
        OgreBites::Slider* backIllumSlider;
        OgreBites::Slider* shadeCoeffSlider;
        OgreBites::Slider* specCoeffSlider;
        OgreBites::Slider* specMultSlider;
        OgreBites::Slider* miscSlider;

};

//|||||||||||||||||||||||||||||||||||||||||||||||

#endif

//|||||||||||||||||||||||||||||||||||||||||||||||
