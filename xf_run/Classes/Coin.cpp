#include "Coin.h"

Coin::Coin(Node* node,XF* xf)
{
	this->xf = xf;
	this->coin = Sprite3D::create("model/coin.c3b");
	this->coin->setRotation3D(Vec3(-90,0,90));
	//this->coin->setPosition3D(Vec3(-10,0,-250));
	this->coin->setScale(0.3);
	//node->addChild(coin,10);
	auto action = new XFCAction(node,xf);
	//CCLOG("node posx ----%f",coin->getPosition3D().x);
	//Vec3 temp = coin->getPosition3D();
	 this->coin->runAction(action);
	//node->setCameraMask(2);
	//CCLOG("node pos ----%f",coin->getPosition3D().x);
}
Sprite3D* Coin::getCoin(){
    return this->coin;
}
void Coin::addCoinToNode(Node* node){
	node->addChild(this->coin,10);
	node->setCameraMask((unsigned short )CameraFlag::USER1);
}
