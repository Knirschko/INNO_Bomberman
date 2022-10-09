// Copyright Epic Games, Inc. All Rights Reserved.

#include "INNO_BombermanGameMode.h"
#include "INNO_BombermanPlayerController.h"
#include "INNO_BombermanCharacter.h"
#include "UObject/ConstructorHelpers.h"

AINNO_BombermanGameMode::AINNO_BombermanGameMode()
{
	// use our custom PlayerController class
	PlayerControllerClass = AINNO_BombermanPlayerController::StaticClass();

	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/TopDown/Blueprints/BP_TopDownCharacter"));
	if (PlayerPawnBPClass.Class != nullptr)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}

	// set default controller to our Blueprinted controller
	static ConstructorHelpers::FClassFinder<APlayerController> PlayerControllerBPClass(TEXT("/Game/TopDown/Blueprints/BP_TopDownPlayerController"));
	if(PlayerControllerBPClass.Class != NULL)
	{
		PlayerControllerClass = PlayerControllerBPClass.Class;
	}
}