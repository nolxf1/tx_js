#include "XFRunGameScene.h"
#include "XFRunGameStart.h"
Scene* XFRunGameScene::createScene(){
	auto scene = Scene::create();
	auto camera = Camera::createOrthographic(960,480,0,1000);
	camera->setCameraFlag(CameraFlag::USER2);
	auto bgLayer = Layer::create();
	scene->addChild(camera);
	scene->addChild(bgLayer,-100);
	auto backGround = Sprite::create("model/back_ground.jpg");
	backGround->setAnchorPoint(Vec2(0,0));
	bgLayer->addChild(backGround);
	bgLayer->setCameraMask(4);
	auto gameLayer = XFRunGameScene::create();
	scene->addChild(gameLayer,100);
	return scene;
}
void XFRunGameScene::loadSource(){
     SimpleAudioEngine::getInstance()->preloadEffect("getcoin.mp3");
}
bool XFRunGameScene::init(){
	if(!Layer::init()){
	   return false;
	}
   //load source
   loadSource();
  //create xf
   xf = new XF();
   this->addChild(xf->getXF(),10);
   xf->getXF()->setPosition3D(Vec3(0,0,-40));
   //create gamelayer camera
   auto s = Director::getInstance()->getWinSize();
   auto camera =Camera::createPerspective(60, (GLfloat)s.width/s.height, 1, 200);
   camera->setCameraFlag(CameraFlag::USER1);
   camera->setPosition3D(Vec3(0, 15, -15));
   camera->lookAt(Vec3(0,0,-60),Vec3(0,1,0));
   this->addChild(camera);
   xf->getXF()->setCameraMask((unsigned short )CameraFlag::USER1);
   //create score
   score = 0;
   lable =  Label::createWithTTF("score:0", "fonts/Marker Felt.ttf", 44);
   lable->setPosition(s.width/2,s.height-20);
   //create item
   auto quitl = Label::createWithTTF("quit", "fonts/Marker Felt.ttf", 44);
   auto  quit = MenuItemLabel::create(quitl, CC_CALLBACK_1(XFRunGameScene::quit, this));
   auto menuq = Menu::create(quit,nullptr);
   menuq->setPosition(50,40);
   this->addChild(menuq);
   auto fastl = Label::createWithTTF("fast", "fonts/Marker Felt.ttf", 44);
   fast = MenuItemLabel::create(fastl, CC_CALLBACK_1(XFRunGameScene::skill, this));
   auto menuf = Menu::create(fast,nullptr);
   menuf->setPosition(s.width-50,40);
   this->addChild(menuf);
   this->fast->setEnabled(false);
   this->fast->setColor(ccc3(0,0,0));
   this->addChild(lable,100);
   //listen touch
   tc = new TouchControl(xf);
   auto touchListener = EventListenerTouchOneByOne::create();
   touchListener->onTouchBegan=CC_CALLBACK_2(XFRunGameScene::onTouchBegan,this);
   touchListener->onTouchEnded=CC_CALLBACK_2(XFRunGameScene::onTouchEnded,this);
   _eventDispatcher->addEventListenerWithSceneGraphPriority(touchListener,this);
     //add map
   mc = new MapControl(this,xf);
   mc->preS();
   schedule(schedule_selector(XFRunGameScene::upDateScene));
   return true;
}
bool XFRunGameScene::onTouchBegan(Touch *touch, Event *unused_event)
{
    tc->touchBegin(touch->getLocation());
	return true;
}

void XFRunGameScene::onTouchEnded(Touch *touch, Event *unused_event)
{
    tc->touchEnd(touch->getLocation());
}
void XFRunGameScene::upDateScene(float dt){
	mc->generateMap(dt);
	if(this->xf->getXFSkill()==1){
		this->xf->setSkillTime(this->xf->getSkillTime()-dt);
	}
	if(this->xf->getSkillTime()<=0&&this->xf->getXFSkill()==1){
	  
	 this->xf->setSkill(0);//取消当前技能状态
     this->xf->stopSkill();
	}
	if(this->score>=200&&this->xf->getXFSkill()==0){
	  this->fast->setEnabled(true);
	  auto color_action = TintBy::create(2.f, 0, -255, -255);
      auto color_back = color_action->reverse();
      auto seq = Sequence::create(color_action, color_back, nullptr);
	  fast->runAction(RepeatForever::create(seq));
	}
}
void XFRunGameScene::getCoin(){
	 score+=100;
	 char temp[100];
	 sprintf(temp,"score:%d",score);
	 lable->setString(temp);
	 SimpleAudioEngine::getInstance()->playEffect("getcoin.mp3", false);
}
void XFRunGameScene::quit(Ref* pSender){
	this->unscheduleUpdate();
	 auto t = TransitionFade::create(0.5f,XFRunGameStart::createScene());
	Director::getInstance()->replaceScene(t);
}
void XFRunGameScene::skill(Ref* pSender){
	this->xf->setSkill(1);//开启技能
	this->xf->setSkillTime(10);//设置技能时间
	xf->runSkill();
	this->fast->setEnabled(false);
	this->fast->stopAllActions();
	score-=200;
	this->fast->setColor(ccc3(0,0,0));
	char temp[100];
	sprintf(temp,"score:%d",score);
	lable->setString(temp);

}
void XFRunGameScene::gameover(){
	if(xf->getXFSkill()!=1){
	this->unscheduleUpdate();
	SimpleAudioEngine::getInstance()->pauseAllEffects();
    auto t = TransitionFade::create(0.5f,XFRunGameStart::createScene());
	Director::getInstance()->replaceScene(t);
	}
}