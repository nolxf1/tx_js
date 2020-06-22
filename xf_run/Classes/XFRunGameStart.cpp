#include "XFRunGameStart.h"
#include "XFRunGameScene.h"
Scene* XFRunGameStart::createScene(){
	auto scene = Scene::create();
	auto layer = XFRunGameStart::create();
	scene->addChild(layer);
	return scene;
}
bool XFRunGameStart::init(){
	if(!Layer::init()){
	   return false;
	}
	auto s = Director::getInstance()->getWinSize();
	auto * pageView = PageView::create();  //新建
	pageView->setSize(Size(s.width, s.height-40));  //设置大小

	Sprite3D* xf1 = Sprite3D::create("model/zhanshi_pao.c3b");
	xf1->setAnchorPoint(Vec2(0.5,0.5));
	xf1->setPosition(Vec2(s.width/2,(s.height)/2));
	xf1->setScale(50);
	Layout* layer1 = Layout::create();
	layer1->addChild(xf1,10);
	layer1->setAnchorPoint(Vec2(0,0));
	layer1->setPosition(Vec2(0,0));
	auto aniamtionly1 =Animation3D::create("model/zhanshi_pao.c3b");
	auto animately1 =Animate3D::create(aniamtionly1);
	xf1->runAction(RepeatForever::create(animately1));


	Sprite3D* xx = Sprite3D::create("Sprite3DTest/ReskinGirl.c3b");
	xx->setAnchorPoint(Vec2(0.5,0.5));
	xx->setPosition(s.width/2,(s.height)/2);
	xx->setScale(8);
	Layout* layer2 = Layout::create();
	layer2->addChild(xx,10);
	layer2->setAnchorPoint(Vec2(0,0));
	layer2->setPosition(Vec2(0,0));
	auto aniamtionly2 =Animation3D::create("Sprite3DTest/ReskinGirl.c3b");
	auto animately2 =Animate3D::create(aniamtionly2);
	CCSpeed* pSpeedly2= CCSpeed::create(RepeatForever::create(animately2), 2.f); //2倍速运行
	xx->runAction(pSpeedly2);

	Sprite3D* xx1 = Sprite3D::create("Sprite3DTest/girl.c3b");
	xx1->setAnchorPoint(Vec2(0.5,0.5));
	xx1->setPosition(s.width/2,(s.height)/2);
	xx1->setScale(2.5);
	Layout* layer3 = Layout::create();
	layer3->addChild(xx1,10);
	layer3->setAnchorPoint(Vec2(0,0));
	layer3->setPosition(Vec2(0,0));
	auto aniamtion =Animation3D::create("Sprite3DTest/girl.c3b");
	auto animate =Animate3D::create(aniamtion);
	CCSpeed* pSpeed= CCSpeed::create(RepeatForever::create(animate), 2.f); //1.5倍速运行
	xx1->runAction(pSpeed);

	pageView->addPage(layer1);
	pageView->addPage(layer3);
	pageView->addPage(layer2);
	pageView->setPosition(Vec2(0,40));
	this->addChild(pageView);

	auto label = Label::createWithTTF("start", "fonts/Marker Felt.ttf", 44);
	auto item = MenuItemLabel::create(label, CC_CALLBACK_1(XFRunGameStart::start, this));
	auto menu = Menu::create(item,nullptr);
	menu->setPosition(Vec2(s.width/2,30));
	this->addChild(menu);
	return true;
}
void XFRunGameStart::start(Ref* pSender){
	auto t = TransitionFade::create(0.5f,XFRunGameScene::createScene());
	Director::getInstance()->replaceScene(t);
}