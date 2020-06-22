#ifndef XF_H
#define XF_H
#include"cocos2d.h"
using namespace cocos2d;
class XF
{
public:
  XF();
  Sprite3D* getXF();
  int getXFSkill();
  void setSkill(int skill);
  float getSkillTime();
  void setSkillTime(float time);
  void runSkill();
  void stopSkill();
private:
	Sprite3D* xf;
	int skill;//默认为0，就是没有技能，，如果为1的时候就是表示有技能
	float skillTime;
	RepeatForever* r;
};
#endif

