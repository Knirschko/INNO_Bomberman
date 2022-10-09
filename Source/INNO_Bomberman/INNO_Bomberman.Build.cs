// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class INNO_Bomberman : ModuleRules
{
	public INNO_Bomberman(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay", "NavigationSystem", "AIModule", "Niagara" });
    }
}
