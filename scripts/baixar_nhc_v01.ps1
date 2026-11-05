param(
    [Parameter(Mandatory=$true)][string]$IndexAtl,
    [Parameter(Mandatory=$true)][string]$IndexEpac,
    [Parameter(Mandatory=$true)][string]$OutRoot,
    [Parameter(Mandatory=$true)][string]$LogPath
)
# Baixa os PDFs do NHC TCR 2025 (somente hosts autorizados), sequencial, com pausa de 1 a 2 s.
# Nao abre, nao executa e nao altera os PDFs depois de baixados.
$ErrorActionPreference = 'Continue'
$primary = 'https://prod-east-nhc.woc.noaa.gov'
$fallback = 'https://www.nhc.noaa.gov'
$rand = New-Object System.Random

function Get-Links($html) {
    [regex]::Matches((Get-Content $html -Raw), 'href="/data/tcr/([A-Za-z0-9_]+\.pdf)"') |
        ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
}

function Try-Download($url, $dest) {
    $part = "$dest.part"
    if (Test-Path $part) { Remove-Item $part -Force }
    & curl.exe -f -sS -L --connect-timeout 30 --max-time 300 -o $part $url
    $code = $LASTEXITCODE
    if ($code -eq 0 -and (Test-Path $part)) {
        $fs = [System.IO.File]::OpenRead($part)
        $buf = New-Object byte[] 5
        [void]$fs.Read($buf, 0, 5)
        $fs.Close()
        $magic = [System.Text.Encoding]::ASCII.GetString($buf)
        if ($magic -ne '%PDF-') { $code = 999 }
    }
    if ($code -eq 0) {
        Move-Item $part $dest
        return 0
    }
    if (Test-Path $part) { Remove-Item $part -Force }
    return $code
}

$jobs = @()
foreach ($f in (Get-Links $IndexAtl))  { $jobs += [pscustomobject]@{ basin='atl';  file=$f } }
foreach ($f in (Get-Links $IndexEpac)) { $jobs += [pscustomobject]@{ basin='epac'; file=$f } }
"Total de links: $($jobs.Count)"

foreach ($j in $jobs) {
    $dir = Join-Path $OutRoot $j.basin
    $dest = Join-Path $dir $j.file
    if (Test-Path $dest) {
        "PULA (ja existe): $dest"
        continue
    }
    $attempts = @(
        "$primary/data/tcr/$($j.file)",
        "$primary/data/tcr/$($j.file)",
        "$fallback/data/tcr/$($j.file)"
    )
    $ok = $false
    $n = 0
    $usedUrl = $null
    $lastCode = $null
    foreach ($u in $attempts) {
        $n++
        $code = Try-Download $u $dest
        $lastCode = $code
        if ($code -eq 0) { $ok = $true; $usedUrl = $u; break }
        "  falha tentativa $n ($u) curl_exit=$code"
        Start-Sleep -Milliseconds (1000 + $rand.Next(0, 1000))
    }
    $utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $rec = [ordered]@{
        basin = $j.basin; file = $j.file; ok = $ok; source_url = $usedUrl
        download_utc = $utc; attempts = $n; last_curl_exit = $lastCode
    }
    ($rec | ConvertTo-Json -Compress) | Add-Content -Path $LogPath -Encoding utf8
    if ($ok) { "OK  $($j.basin) $($j.file) via $usedUrl (tentativa $n)" } else { "ERRO $($j.basin) $($j.file)" }
    Start-Sleep -Milliseconds (1000 + $rand.Next(0, 1000))
}
"Fim"
