#ifndef COIN_H
#define COIN_H
#include"cocos2d.h"
#include"XF.h"
#include"XFCAction.h"
using namespace cocos2d;
class Coin
{
public:
	Coin(Node* node,XF* xf);
	Sprite3D* getCoin();
	void addCoinToNode(Node* node);
private:
	Sprite3D* coin;
	XF* xf;
};
#endif
