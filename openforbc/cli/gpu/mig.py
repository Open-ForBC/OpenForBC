from typing import Optional

from typer import Context, Option, echo, Typer  # noqa: TC002

from openforbc.cli.gpu.state import get_gpu_uuid
from openforbc.cli.state import get_api_client, state as global_state

mig = Typer(help="Manage NVIDIA MIG GPUs")


mig_gi = Typer(help="Manage MIG GPU instances")
mig.add_typer(mig_gi, name="gi")


@mig_gi.callback(invoke_without_command=True)
def mig_gi_callback(ctx: Context) -> None:
    if ctx.invoked_subcommand is None:
        return ctx.invoke(list_mig_gpu_instances)


@mig_gi.command("profiles")
def list_gpu_instance_profiles() -> None:
    client = get_api_client()
    gpu_uuid = get_gpu_uuid()

    profiles = client.get_gi_profiles(gpu_uuid)
    for profile in profiles:
        capacity = client.get_gpu_instance_profile_capacity(gpu_uuid, profile.id)
        echo(
            f"{profile.id}: {profile} (slices={profile.slice_count}, "
            f"memory={profile.memory_size}MB) ({capacity} available)"
        )


@mig_gi.command("create")
def create_gpu_instance(
    gip_id: int,
    default_ci: bool = Option(
        False, "--default-ci", "-d", help="Also create default compute instance"
    ),
    id_only: bool = Option(False, "--id-only", "-q"),
) -> None:
    instance = get_api_client().create_gpu_instance(get_gpu_uuid(), gip_id, default_ci)
    echo(instance.id if id_only else instance)


@mig_gi.command("list")
def list_mig_gpu_instances() -> None:
    client = global_state["api_client"]
    gpu_uuid = get_gpu_uuid()

    instances = client.get_gpu_instances(gpu_uuid)
    for instance in instances:
        echo(instance)


@mig_gi.command("destroy")
def destroy_gpu_instance(instance_id: int) -> None:
    client = global_state["api_client"]
    gpu_uuid = get_gpu_uuid()

    client.destroy_gpu_instance(gpu_uuid, instance_id)


mig_ci = Typer(help="Manage MIG compute instances")
mig.add_typer(mig_ci, name="ci")


@mig_ci.callback(invoke_without_command=True)
def mig_ci_callback(ctx: Context) -> None:
    if ctx.invoked_subcommand is None:
        return ctx.invoke(list_compute_instances, None)


@mig_ci.command("list")
def list_compute_instances(
    gi_id: Optional[int] = Option(None, "--gi-id", "-gi")
) -> None:
    """List all compute instances (or filter a specific GPU instance)."""
    client = get_api_client()
    gpu_uuid = get_gpu_uuid()

    for instance in (
        client.get_compute_instances(gpu_uuid)
        if gi_id is None
        else client.get_gi_compute_instances(gpu_uuid, gi_id)
    ):
        echo(instance)


@mig_ci.command("profiles")
def list_compute_instance_profiles(gi_id: int = Option(..., "--gi-id", "-gi")) -> None:
    client = get_api_client()
    gpu_uuid = get_gpu_uuid()
    for profile in client.get_compute_instance_profiles(gpu_uuid, gi_id):
        capacity = client.get_compute_instance_profile_capacity(
            gpu_uuid, gi_id, profile.id
        )
        echo(f"{profile.id}: {profile} ({capacity} available)")


@mig_ci.command("create")
def create_compute_instance(
    cip_id: int, gi_id: int = Option(..., "--gi-id", "-gi")
) -> None:
    ci = get_api_client().create_compute_instance(get_gpu_uuid(), gi_id, cip_id)
    echo(ci)


@mig_ci.command("destroy")
def destroy_compute_instance(
    ci_id: int, gi_id: int = Option(..., "--gi-id", "-gi")
) -> None:
    get_api_client().destroy_compute_instance(get_gpu_uuid(), gi_id, ci_id)
