#include "XF.h"

XF::XF(){
	this->xf = Sprite3D::create("model/zhanshi_pao.c3b"); 

	this->xf->setRotation3D(Vec3(0,165,0));
	
	auto aniamtion =Animation3D::create("model/zhanshi_pao.c3b");
	auto animate =Animate3D::create(aniamtion);
	this->xf->runAction(RepeatForever::create (animate));
	this->skill = 0;
	this->skillTime = 0;
}
Sprite3D* XF::getXF(){
	return this->xf;
}
int XF::getXFSkill(){
	return this->skill;
}
void XF::setSkill(int skill){
	this->skill = skill;
}
float XF::getSkillTime(){
    return this->skillTime;
}
void XF::setSkillTime(float time){
	this->skillTime = time;
}
void XF::runSkill(){
	auto act = MoveTo::create(0.2,Vec3(xf->getPositionX(),xf->getPositionY(),-30));
	this->xf->runAction(act);
	auto color_action = TintBy::create(0.5f, 0, -244, -244);
     auto color_back = color_action->reverse();
     auto seq = Sequence::create(color_action, color_back, nullptr);
	 r = RepeatForever::create(seq);
	this->xf->runAction(r);
}
void XF::stopSkill(){
	r->stop();
	auto act = MoveTo::create(0.2,Vec3(xf->getPositionX(),xf->getPositionY(),-40));
	this->xf->runAction(act);
}
