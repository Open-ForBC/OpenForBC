from typer import Context, Option, echo, Typer
from openforbc.cli.gpu.state import get_gpu_uuid

from openforbc.cli.state import get_api_client, state as global_state

mig = Typer(help="Manage NVIDIA MIG GPUs")


mig_gi = Typer(help="Manager MIG GPU instances")
mig.add_typer(mig_gi, name="gi")


@mig_gi.callback(invoke_without_command=True)
def mig_gi_callback(ctx: Context) -> None:
    if ctx.invoked_subcommand is None:
        return ctx.invoke(list_mig_gpu_instances)


@mig_gi.command("profiles")
def list_gpu_instance_profiles() -> None:
    gpu_uuid = get_gpu_uuid()

    profiles = get_api_client().get_mig_profiles(gpu_uuid)
    for profile in profiles:
        echo(
            f"{profile} (slices={profile.slice_count}, memory={profile.memory_size}MB)"
        )


@mig_gi.command("create")
def create_gpu_instance(
    profile_id: int, id_only: bool = Option(False, "--id-only", "-q")
) -> None:
    instance = get_api_client().create_gpu_instance(get_gpu_uuid(), profile_id)
    echo(instance.id if id_only else instance)


@mig_gi.command("list")
def list_mig_gpu_instances() -> None:
    client = global_state["api_client"]
    gpu_uuid = get_gpu_uuid()

    instances = client.get_mig_instances(gpu_uuid)
    for instance in instances:
        echo(instance)


@mig_gi.command("destroy")
def destroy_gpu_instance(instance_id: int) -> None:
    client = global_state["api_client"]
    gpu_uuid = get_gpu_uuid()

    client.destroy_gpu_instance(gpu_uuid, instance_id)
