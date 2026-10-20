# 子库操作 / Submodule guide

## 中文

### 版本关联方式

`vaws_wd` 通过 Git submodule 引用子库：

| 项目 | 配置 |
| --- | --- |
| 子库路径 | `aisbench_auto_tools_prefix` |
| 子库地址 | `https://github.com/dianWang325/aisbench_auto_tools_prefix.git` |
| 更新跟踪分支 | `dev/wd_test` |

主库提交记录的是子库的**具体提交 SHA**。`.gitmodules` 中的分支配置用于查找更新；子库分支更新后，主库关联版本不会自动变化。以下命令除首次克隆外，均在 `vaws_wd` 根目录执行。

### 1. 新机器首次拉取

同时克隆主库及子库：

```bash
git clone --recurse-submodules https://github.com/dianWang325/vaws_wd.git
cd vaws_wd
```

如果主库已经克隆，补充下载子库：

```bash
git submodule update --init --recursive
```

### 2. 日常同步已关联的版本

```bash
git pull --ff-only
git submodule update --init --recursive
```

子库将切换到主库记录的提交，使不同机器使用相同版本。这一步不会主动采用子库分支的最新提交。

### 3. 采用子库最新版本，并记录到主库

开始前检查主库和子库的本地修改，先提交或妥善保存需要保留的内容。

```bash
git status --short
git -C aisbench_auto_tools_prefix status --short
git pull --ff-only

# 获取并检出 dev/wd_test 的最新提交
git submodule update --init --remote aisbench_auto_tools_prefix

# 查看新旧子库版本之间的提交，并完成所需功能验证
git diff --submodule=log -- aisbench_auto_tools_prefix

# 将新的子库提交指针记录到主库
git config user.name dianWang325
git config user.email 93850926+dianWang325@users.noreply.github.com
git add aisbench_auto_tools_prefix
git commit -s -m "Update aisbench_auto_tools_prefix submodule"

# 推送前核对作者、提交者和 Signed-off-by
git log -1 --format=full
git push origin main
```

作者、提交者和 `Signed-off-by` 必须均为 `dianWang325 <93850926+dianWang325@users.noreply.github.com>`。如没有版本变化，无需创建提交。其他机器随后执行第 2 节的命令即可同步新的关联版本。

### 4. 直接在子库中开发

按固定提交初始化后，子库处于 detached HEAD 是正常现象。开发前切换到分支：

```bash
git -C aisbench_auto_tools_prefix fetch origin
git -C aisbench_auto_tools_prefix switch dev/wd_test
git -C aisbench_auto_tools_prefix pull --ff-only origin dev/wd_test
git -C aisbench_auto_tools_prefix config user.name dianWang325
git -C aisbench_auto_tools_prefix config user.email 93850926+dianWang325@users.noreply.github.com
```

如果本地尚无该分支，使用下面的命令代替 `switch dev/wd_test`：

```bash
git -C aisbench_auto_tools_prefix switch --track -c dev/wd_test origin/dev/wd_test
```

完成修改后，在子库中暂存指定文件、使用 `git commit -s` 提交，并用 `git log -1 --format=full` 核对上述身份。**先推送子库提交**：

```bash
git -C aisbench_auto_tools_prefix push origin dev/wd_test
```

然后在主库中执行第 3 节的配置身份、`git add`、提交、身份核对和推送步骤。这样主库引用的子库提交才能被其他机器下载。仅推送子库不会更新主库记录的版本。

## English

### How versions are linked

`vaws_wd` includes `https://github.com/dianWang325/aisbench_auto_tools_prefix.git` as a Git submodule at `aisbench_auto_tools_prefix`. Its configured update branch is `dev/wd_test`.

Each parent repository commit pins an **exact submodule commit SHA**. The branch setting in `.gitmodules` tells Git where to look for updates; it does not automatically advance the pinned version. Run the following commands from the `vaws_wd` root, except for the initial clone.

### 1. Clone on a new machine

Clone the parent repository and its submodules together:

```bash
git clone --recurse-submodules https://github.com/dianWang325/vaws_wd.git
cd vaws_wd
```

If the parent repository is already cloned, initialize and download its submodules:

```bash
git submodule update --init --recursive
```

### 2. Synchronize the pinned versions

```bash
git pull --ff-only
git submodule update --init --recursive
```

This checks out the submodule commit recorded by the parent repository so machines use the same version. It does not select the latest commit on the submodule branch.

### 3. Adopt the latest submodule version

Check both repositories for local changes first. Commit or safely save changes you need to keep before updating.

```bash
git status --short
git -C aisbench_auto_tools_prefix status --short
git pull --ff-only

# Fetch and check out the latest commit on dev/wd_test
git submodule update --init --remote aisbench_auto_tools_prefix

# Review the commits between the old and new versions; validate functionality
git diff --submodule=log -- aisbench_auto_tools_prefix

# Record the new submodule commit pointer in the parent repository
git config user.name dianWang325
git config user.email 93850926+dianWang325@users.noreply.github.com
git add aisbench_auto_tools_prefix
git commit -s -m "Update aisbench_auto_tools_prefix submodule"

# Verify the author, committer and Signed-off-by before pushing
git log -1 --format=full
git push origin main
```

The author, committer and `Signed-off-by` must all be `dianWang325 <93850926+dianWang325@users.noreply.github.com>`. No commit is needed if the pinned version has not changed. Other machines can then run the commands in section 2 to obtain the updated version.

### 4. Develop directly in the submodule

A detached HEAD is normal when checking out a pinned commit. Switch to the development branch before making commits:

```bash
git -C aisbench_auto_tools_prefix fetch origin
git -C aisbench_auto_tools_prefix switch dev/wd_test
git -C aisbench_auto_tools_prefix pull --ff-only origin dev/wd_test
git -C aisbench_auto_tools_prefix config user.name dianWang325
git -C aisbench_auto_tools_prefix config user.email 93850926+dianWang325@users.noreply.github.com
```

If the local branch does not exist, replace `switch dev/wd_test` with:

```bash
git -C aisbench_auto_tools_prefix switch --track -c dev/wd_test origin/dev/wd_test
```

Stage the intended files inside the submodule, commit with `git commit -s`, and verify the required identity with `git log -1 --format=full`. **Push the submodule commit first**:

```bash
git -C aisbench_auto_tools_prefix push origin dev/wd_test
```

Then configure the parent repository identity, stage the submodule pointer, commit, verify the identity and push as shown in section 3. This ensures other machines can download the referenced commit. Pushing the submodule alone does not change the version pinned by the parent repository.
