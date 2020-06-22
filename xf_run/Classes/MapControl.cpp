#include "MapControl.h"
#include "XFCAction.h"
#include "XFRunTAction.h"
#include "XFOAction.h"
MapControl::MapControl(Node* node,XF* xf)
{
	this->node = node;
	this->xf = xf;
}


MapControl::~MapControl(void)
{
}

void MapControl::generateMap(float dt){
	 static float sstep = 0;
	 static float cstep = 0;
	 static float ostep = 0;
	 static float tstep = 0;
	 sstep+=dt;
	 cstep+=dt;
	 if(sstep>=3.87){
		 sstep = 0;
		this->generateS();
	 }
	 if(cstep>0.7){
	   cstep=0;
	   this->generateC();
	 }
	 tstep+=dt;
	 //1.1
	 if(tstep>1.1){
		 tstep = 0;
		 this->generateT();
	 }
	 ostep+=dt;
	 if(ostep>0.9){
	    ostep = 0;
		this->generateO();
	 }
}
//建造跑道
void MapControl::generateS(){
	Sprite3D::createAsync("model/scene.c3b", CC_CALLBACK_2(MapControl::callback_S, this),node);
}
void MapControl::callback_S(Sprite3D* ss, void* param){
	Node* node = (Node*)param;
	ss->setScale(0.2); 
    ss->setRotation3D(Vec3(0,90,0));
	node->addChild(ss,10); 
	auto action = new XFSAction();
	ss->setPosition3D(Vec3(0,-5,-550));  
	ss->runAction(action); 
	ss->setCameraMask((unsigned short )CameraFlag::USER1);
}
//创建金币
void MapControl::generateC(){
	Sprite3D::createAsync("model/coin.c3b", CC_CALLBACK_2(MapControl::callback_C, this),node);
}
void MapControl::callback_C(Sprite3D* coin, void* param){
	Node* node = (Node*)param;
	coin->setScale(0.3);
	coin->setRotation3D(Vec3(90,180,180));
	node->addChild(coin,10);
	int random = (int)(CCRANDOM_0_1() * 2.9); //[0,100]   CCRANDOM_0_1() 取值范围是[0,1]
	coin->setPosition3D(Vec3 (-10+10*(random),0,-250));
	coin->runAction(new XFCAction(node,xf));
	coin->setCameraMask(2);
}
//创建障碍
void MapControl::generateO(){
	Sprite3D::createAsync("model/piglet.c3b", CC_CALLBACK_2(MapControl::callback_O, this),node);
}
void MapControl::callback_O(Sprite3D* o,void* param){
	o->setTexture("model/zhu0928.jpg");
	o->setScale(0.1);
	auto action = new XFOAction(node,xf);
	o->runAction(action);
    
    auto aniamtion =Animation3D::create("model/piglet.c3b");
    auto animate =Animate3D::createWithFrames(aniamtion,135,147);
    o->runAction(RepeatForever::create (animate));
	node->addChild(o,10);
	o->setCameraMask(2);
	int random = (int)(CCRANDOM_0_1() * 2.9); //[0,100]   CCRANDOM_0_1() 取值范围是[0,1]
	o->setPosition3D(Vec3 (-10+10*(random),-5,-250));
}
//
void MapControl::generateT(){
    std::string fileName = "model/tortoise.c3b";
	Sprite3D::createAsync(fileName, CC_CALLBACK_2(MapControl::callback_T, this),node);
}
void MapControl::callback_T(Sprite3D* ts, void* param){
	Node* node = (Node*)param;
	ts->setScale(0.005);
	int random = (int)(CCRANDOM_0_1() * 2.9); //[0,100]   CCRANDOM_0_1() 取值范围是[0,1]
    ts->setPosition3D(Vec3 (-10+10*(random),0,-250));
    node->addChild(ts,10);
    auto animation = Animation3D::create("model/tortoise.c3b");
    if (animation)
    {
        auto animate = Animate3D::create(animation, 0.f, 1.933f);
		auto _swim = RepeatForever::create(animate);
        ts->runAction(_swim);
		XFRunTAction* xfrta = new XFRunTAction();
		ts->runAction(xfrta);
	}else{
		ts->removeFromParent();
		ts = nullptr;
	}
	ts->setRotation3D(Vec3(0,90,0));
	ts->setCameraMask(2);
}
void MapControl::preS(){
	auto scene_1 =cocos2d::Sprite3D::create("model/scene.c3b");  
    scene_1->setScale(0.2); 
    scene_1->setRotation3D(Vec3(0,90,0));
    node->addChild(scene_1,10); 
    auto action_1 = new XFSAction();
    scene_1->setPosition3D(Vec3(0,-5,0));  
    scene_1->runAction(action_1); 
	scene_1->setCameraMask(2);

    auto scene_2 =cocos2d::Sprite3D::create("model/scene.c3b");  
    scene_2->setScale(0.2); 
    scene_2->setRotation3D(Vec3(0,90,0));
    node->addChild(scene_2,10); 
    auto action_2 = new XFSAction();
    scene_2->setPosition3D(Vec3(0,-5,-470));  
    scene_2->runAction(action_2); 
	scene_2->setCameraMask(2);
}